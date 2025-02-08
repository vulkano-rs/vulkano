use crate::{
    buffer::{BufferContents, BufferUsage, Subbuffer},
    command_buffer::{
        auto::Resource, sys::RecordingCommandBuffer, AutoCommandBufferBuilder, ResourceInCommand,
    },
    device::{Device, DeviceOwned, QueueFlags},
    format::{ClearColorValue, ClearDepthStencilValue, FormatFeatures},
    image::{Image, ImageAspects, ImageLayout, ImageSubresourceRange, ImageUsage},
    sync::PipelineStageAccessFlags,
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, SafeDeref, ValidationError, Version,
    VulkanObject,
};
use ash::vk;
use smallvec::{smallvec, SmallVec};
use std::{mem::size_of_val, sync::Arc};

/// # Commands to fill resources with new data.
impl<L> AutoCommandBufferBuilder<L> {
    /// Clears a color image with a specific value.
    pub fn clear_color_image(
        &mut self,
        clear_info: ClearColorImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_color_image(&clear_info)?;

        Ok(unsafe { self.clear_color_image_unchecked(clear_info) })
    }

    fn validate_clear_color_image(
        &self,
        clear_info: &ClearColorImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_clear_color_image(clear_info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdClearColorImage-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_color_image_unchecked(
        &mut self,
        clear_info: ClearColorImageInfo,
    ) -> &mut Self {
        let &ClearColorImageInfo {
            ref image,
            image_layout,
            clear_value: _,
            ref regions,
            _ne: _,
        } = &clear_info;

        self.add_command(
            "clear_color_image",
            regions
                .iter()
                .cloned()
                .flat_map(|subresource_range| {
                    [(
                        ResourceInCommand::Destination.into(),
                        Resource::Image {
                            image: image.clone(),
                            subresource_range,
                            memory_access: PipelineStageAccessFlags::Clear_TransferWrite,
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.clear_color_image_unchecked(&clear_info) };
            },
        );

        self
    }

    /// Clears a depth/stencil image with a specific value.
    pub fn clear_depth_stencil_image(
        &mut self,
        clear_info: ClearDepthStencilImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_depth_stencil_image(&clear_info)?;

        Ok(unsafe { self.clear_depth_stencil_image_unchecked(clear_info) })
    }

    fn validate_clear_depth_stencil_image(
        &self,
        clear_info: &ClearDepthStencilImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_clear_depth_stencil_image(clear_info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdClearDepthStencilImage-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_depth_stencil_image_unchecked(
        &mut self,
        clear_info: ClearDepthStencilImageInfo,
    ) -> &mut Self {
        let &ClearDepthStencilImageInfo {
            ref image,
            image_layout,
            clear_value: _,
            ref regions,
            _ne: _,
        } = &clear_info;

        self.add_command(
            "clear_depth_stencil_image",
            regions
                .iter()
                .cloned()
                .flat_map(|subresource_range| {
                    [(
                        ResourceInCommand::Destination.into(),
                        Resource::Image {
                            image: image.clone(),
                            subresource_range,
                            memory_access: PipelineStageAccessFlags::Clear_TransferWrite,
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.clear_depth_stencil_image_unchecked(&clear_info) };
            },
        );

        self
    }

    /// Fills a region of a buffer with repeated copies of a value.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatedly written through the entire buffer.
    pub fn fill_buffer(
        &mut self,
        dst_buffer: Subbuffer<[u32]>,
        data: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_fill_buffer(&dst_buffer, data)?;

        Ok(unsafe { self.fill_buffer_unchecked(dst_buffer, data) })
    }

    fn validate_fill_buffer(
        &self,
        dst_buffer: &Subbuffer<[u32]>,
        data: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_fill_buffer(dst_buffer, data)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdFillBuffer-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn fill_buffer_unchecked(
        &mut self,
        dst_buffer: Subbuffer<[u32]>,
        data: u32,
    ) -> &mut Self {
        self.add_command(
            "fill_buffer",
            [(
                ResourceInCommand::Destination.into(),
                Resource::Buffer {
                    buffer: dst_buffer.as_bytes().clone(),
                    range: 0..dst_buffer.size(),
                    memory_access: PipelineStageAccessFlags::Clear_TransferWrite,
                },
            )]
            .into_iter()
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.fill_buffer_unchecked(&dst_buffer, data) };
            },
        );

        self
    }

    /// Writes data to a region of a buffer.
    pub fn update_buffer<D, Dd>(
        &mut self,
        dst_buffer: Subbuffer<D>,
        data: Dd,
    ) -> Result<&mut Self, Box<ValidationError>>
    where
        D: BufferContents + ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        self.validate_update_buffer(
            dst_buffer.as_bytes(),
            size_of_val(data.deref()) as DeviceSize,
        )?;

        Ok(unsafe { self.update_buffer_unchecked(dst_buffer, data) })
    }

    fn validate_update_buffer(
        &self,
        dst_buffer: &Subbuffer<[u8]>,
        data_size: DeviceSize,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_update_buffer(dst_buffer, data_size)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_buffer_unchecked<D, Dd>(
        &mut self,
        dst_buffer: Subbuffer<D>,
        data: Dd,
    ) -> &mut Self
    where
        D: BufferContents + ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        self.add_command(
            "update_buffer",
            [(
                ResourceInCommand::Destination.into(),
                Resource::Buffer {
                    buffer: dst_buffer.as_bytes().clone(),
                    range: 0..size_of_val(data.deref()) as DeviceSize,
                    memory_access: PipelineStageAccessFlags::Clear_TransferWrite,
                },
            )]
            .into_iter()
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.update_buffer_unchecked(&dst_buffer, &data) };
            },
        );

        self
    }
}

impl RecordingCommandBuffer {
    #[inline]
    pub unsafe fn clear_color_image(
        &mut self,
        clear_info: &ClearColorImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_color_image(clear_info)?;

        Ok(unsafe { self.clear_color_image_unchecked(clear_info) })
    }

    fn validate_clear_color_image(
        &self,
        clear_info: &ClearColorImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdClearColorImage-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        clear_info
            .validate(self.device())
            .map_err(|err| err.add_context("clear_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_color_image_unchecked(
        &mut self,
        clear_info: &ClearColorImageInfo,
    ) -> &mut Self {
        if clear_info.regions.is_empty() {
            return self;
        }

        let clear_info_vk = clear_info.to_vk();
        let ranges_vk = clear_info.to_vk_ranges();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_clear_color_image)(
                self.handle(),
                clear_info_vk.image,
                clear_info_vk.image_layout,
                &clear_info_vk.color,
                ranges_vk.len() as u32,
                ranges_vk.as_ptr(),
            )
        };

        self
    }

    #[inline]
    pub unsafe fn clear_depth_stencil_image(
        &mut self,
        clear_info: &ClearDepthStencilImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_depth_stencil_image(clear_info)?;

        Ok(unsafe { self.clear_depth_stencil_image_unchecked(clear_info) })
    }

    fn validate_clear_depth_stencil_image(
        &self,
        clear_info: &ClearDepthStencilImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdClearDepthStencilImage-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        clear_info
            .validate(self.device())
            .map_err(|err| err.add_context("clear_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_depth_stencil_image_unchecked(
        &mut self,
        clear_info: &ClearDepthStencilImageInfo,
    ) -> &mut Self {
        if clear_info.regions.is_empty() {
            return self;
        }

        let clear_info_vk = clear_info.to_vk();
        let ranges_vk = clear_info.to_vk_ranges();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_clear_depth_stencil_image)(
                self.handle(),
                clear_info_vk.image,
                clear_info_vk.image_layout,
                &clear_info_vk.depth_stencil,
                ranges_vk.len() as u32,
                ranges_vk.as_ptr(),
            )
        };

        self
    }

    #[inline]
    pub unsafe fn fill_buffer(
        &mut self,
        dst_buffer: &Subbuffer<[u32]>,
        data: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_fill_buffer(dst_buffer, data)?;

        Ok(unsafe { self.fill_buffer_unchecked(dst_buffer, data) })
    }

    fn validate_fill_buffer(
        &self,
        dst_buffer: &Subbuffer<[u32]>,
        _data: u32,
    ) -> Result<(), Box<ValidationError>> {
        let device = self.device();
        let queue_family_properties = self.queue_family_properties();

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if !queue_family_properties
                .queue_flags
                .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
            {
                return Err(Box::new(ValidationError {
                    problem: "the queue family of the command buffer does not support \
                        transfer, graphics or compute operations"
                        .into(),
                    vuids: &["VUID-vkCmdFillBuffer-commandBuffer-cmdpool"],
                    ..Default::default()
                }));
            }
        } else {
            if !queue_family_properties
                .queue_flags
                .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
            {
                return Err(Box::new(ValidationError {
                    problem: "the queue family of the command buffer does not support \
                        graphics or compute operations"
                        .into(),
                    vuids: &["VUID-vkCmdFillBuffer-commandBuffer-00030"],
                    ..Default::default()
                }));
            }
        }

        // VUID-vkCmdFillBuffer-commonparent
        assert_eq!(device, dst_buffer.device());

        // VUID-vkCmdFillBuffer-size-00026
        // Guaranteed by `Subbuffer`

        if !dst_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_DST)
        {
            return Err(Box::new(ValidationError {
                context: "dst_buffer.buffer().usage()".into(),
                problem: "does not contain `BufferUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-vkCmdFillBuffer-dstBuffer-00029"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdFillBuffer-dstOffset-00024
        // VUID-vkCmdFillBuffer-size-00027
        // Guaranteed by `Subbuffer`

        // VUID-vkCmdFillBuffer-dstOffset-00025
        // VUID-vkCmdFillBuffer-size-00028
        // Guaranteed because we take `Subbuffer<[u32]>`

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn fill_buffer_unchecked(
        &mut self,
        dst_buffer: &Subbuffer<[u32]>,
        data: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_fill_buffer)(
                self.handle(),
                dst_buffer.buffer().handle(),
                dst_buffer.offset(),
                dst_buffer.size(),
                data,
            )
        };

        self
    }

    #[inline]
    pub unsafe fn update_buffer<D>(
        &mut self,
        dst_buffer: &Subbuffer<D>,
        data: &D,
    ) -> Result<&mut Self, Box<ValidationError>>
    where
        D: BufferContents + ?Sized,
    {
        if size_of_val(data) == 0 {
            return Ok(self);
        }

        self.validate_update_buffer(dst_buffer.as_bytes(), size_of_val(data) as DeviceSize)?;

        Ok(unsafe { self.update_buffer_unchecked(dst_buffer, data) })
    }

    fn validate_update_buffer(
        &self,
        dst_buffer: &Subbuffer<[u8]>,
        data_size: DeviceSize,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    transfer, graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdUpdateBuffer-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        // VUID-vkCmdUpdateBuffer-commonparent
        assert_eq!(device, dst_buffer.device());

        // VUID-vkCmdUpdateBuffer-dataSize-arraylength
        // Ensured because we return when the size is 0.

        if !dst_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_DST)
        {
            return Err(Box::new(ValidationError {
                context: "dst_buffer.buffer().usage()".into(),
                problem: "does not contain `BufferUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-dstBuffer-00034"],
                ..Default::default()
            }));
        }

        if data_size > dst_buffer.size() {
            return Err(Box::new(ValidationError {
                problem: "the size of `data` is greater than `dst_buffer.size()`".into(),
                vuids: &[
                    "VUID-vkCmdUpdateBuffer-dstOffset-00032",
                    "VUID-vkCmdUpdateBuffer-dataSize-00033",
                ],
                ..Default::default()
            }));
        }

        if dst_buffer.offset() % 4 != 0 {
            return Err(Box::new(ValidationError {
                context: "dst_buffer.offset()".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-dstOffset-00036"],
                ..Default::default()
            }));
        }

        if data_size > 65536 {
            return Err(Box::new(ValidationError {
                context: "data".into(),
                problem: "the size is greater than 65536 bytes".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-dataSize-00037"],
                ..Default::default()
            }));
        }

        if data_size % 4 != 0 {
            return Err(Box::new(ValidationError {
                context: "data".into(),
                problem: "the size is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdUpdateBuffer-dataSize-00038"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_buffer_unchecked<D>(
        &mut self,
        dst_buffer: &Subbuffer<D>,
        data: &D,
    ) -> &mut Self
    where
        D: BufferContents + ?Sized,
    {
        if size_of_val(data) == 0 {
            return self;
        }

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_update_buffer)(
                self.handle(),
                dst_buffer.buffer().handle(),
                dst_buffer.offset(),
                size_of_val(data) as DeviceSize,
                <*const _>::cast(data),
            )
        };

        self
    }
}

/// Parameters to clear a color image.
#[derive(Clone, Debug)]
pub struct ClearColorImageInfo {
    /// The image to clear.
    ///
    /// There is no default value.
    pub image: Arc<Image>,

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
    /// Returns a default `ClearColorImageInfo` with the provided `image`.
    #[inline]
    pub fn new(image: Arc<Image>) -> Self {
        let range = image.subresource_range();

        Self {
            image,
            image_layout: ImageLayout::TransferDstOptimal,
            clear_value: ClearColorValue::Float([0.0; 4]),
            regions: smallvec![range],
            _ne: crate::NonExhaustive(()),
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn image(image: Arc<Image>) -> Self {
        Self::new(image)
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref image,
            image_layout,
            clear_value: _,
            ref regions,
            _ne: _,
        } = self;

        image_layout.validate_device(device).map_err(|err| {
            err.add_context("image_layout")
                .set_vuids(&["VUID-vkCmdClearColorImage-imageLayout-parameter"])
        })?;

        // VUID-vkCmdClearColorImage-commonparent
        assert_eq!(device, image.device().as_ref());

        if !image.usage().intersects(ImageUsage::TRANSFER_DST) {
            return Err(Box::new(ValidationError {
                context: "image.usage()".into(),
                problem: "does not contain `ImageUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-vkCmdClearColorImage-image-00002"],
                ..Default::default()
            }));
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if !image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    context: "image.format_features()".into(),
                    problem: "does not contain `FormatFeatures::TRANSFER_DST`".into(),
                    vuids: &["VUID-vkCmdClearColorImage-image-01993"],
                    ..Default::default()
                }));
            }
        }

        let image_aspects = image.format().aspects();

        if image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            return Err(Box::new(ValidationError {
                context: "image.format()".into(),
                problem: "is a depth/stencil format".into(),
                vuids: &["VUID-vkCmdClearColorImage-image-00007"],
                ..Default::default()
            }));
        }

        if image.format().compression().is_some() {
            return Err(Box::new(ValidationError {
                context: "image.format()".into(),
                problem: "is a compressed format".into(),
                vuids: &["VUID-vkCmdClearColorImage-image-00007"],
                ..Default::default()
            }));
        }

        if image.format().ycbcr_chroma_sampling().is_some() {
            return Err(Box::new(ValidationError {
                context: "image.format()".into(),
                problem: "is a YCbCr format".into(),
                vuids: &["VUID-vkCmdClearColorImage-image-01545"],
                ..Default::default()
            }));
        }

        if !matches!(
            image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "image_layout".into(),
                problem: "is not `ImageLayout::TransferDstOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-vkCmdClearColorImage-imageLayout-01394"],
                ..Default::default()
            }));
        }

        for (region_index, subresource_range) in regions.iter().enumerate() {
            subresource_range
                .validate(device)
                .map_err(|err| err.add_context(format!("regions[{}]", region_index)))?;

            if !image_aspects.contains(subresource_range.aspects) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].aspects` is not a subset of `image.format().aspects()`",
                        region_index
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearColorImage-aspectMask-02498"],
                    ..Default::default()
                }));
            }

            if subresource_range.mip_levels.end > image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].mip_levels.end` is greater than `image.mip_levels()`",
                        region_index
                    )
                    .into(),
                    vuids: &[
                        "VUID-vkCmdClearColorImage-baseMipLevel-01470",
                        "VUID-vkCmdClearColorImage-pRanges-01692",
                    ],
                    ..Default::default()
                }));
            }

            if subresource_range.array_layers.end > image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].array_layers.end` is greater than `image.array_layers()`",
                        region_index
                    )
                    .into(),
                    vuids: &[
                        "VUID-vkCmdClearColorImage-baseArrayLayer-01472",
                        "VUID-vkCmdClearColorImage-pRanges-01693",
                    ],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> ClearColorImageInfoVk {
        let &Self {
            ref image,
            image_layout,
            clear_value,
            regions: _,
            _ne: _,
        } = self;

        ClearColorImageInfoVk {
            image: image.handle(),
            image_layout: image_layout.into(),
            color: clear_value.to_vk(),
        }
    }

    pub(crate) fn to_vk_ranges(&self) -> SmallVec<[vk::ImageSubresourceRange; 8]> {
        self.regions
            .iter()
            .map(ImageSubresourceRange::to_vk)
            .collect()
    }
}

pub(crate) struct ClearColorImageInfoVk {
    pub(crate) image: vk::Image,
    pub(crate) image_layout: vk::ImageLayout,
    pub(crate) color: vk::ClearColorValue,
}

/// Parameters to clear a depth/stencil image.
#[derive(Clone, Debug)]
pub struct ClearDepthStencilImageInfo {
    /// The image to clear.
    ///
    /// There is no default value.
    pub image: Arc<Image>,

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
    /// Returns a default `ClearDepthStencilImageInfo` with the provided `image`.
    #[inline]
    pub fn new(image: Arc<Image>) -> Self {
        let range = image.subresource_range();

        Self {
            image,
            image_layout: ImageLayout::TransferDstOptimal,
            clear_value: ClearDepthStencilValue::default(),
            regions: smallvec![range],
            _ne: crate::NonExhaustive(()),
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn image(image: Arc<Image>) -> Self {
        Self::new(image)
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref image,
            image_layout,
            clear_value,
            ref regions,
            _ne: _,
        } = self;

        image_layout.validate_device(device).map_err(|err| {
            err.add_context("image_layout")
                .set_vuids(&["VUID-vkCmdClearDepthStencilImage-imageLayout-parameter"])
        })?;

        // VUID-vkCmdClearDepthStencilImage-commonparent
        assert_eq!(device, image.device().as_ref());

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if !image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    context: "image.format_features()".into(),
                    problem: "does not contain `FormatFeatures::TRANSFER_DST`".into(),
                    vuids: &["VUID-vkCmdClearDepthStencilImage-image-01994"],
                    ..Default::default()
                }));
            }
        }

        let image_aspects = image.format().aspects();

        if !image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            return Err(Box::new(ValidationError {
                context: "image.format()".into(),
                problem: "is not a depth/stencil format".into(),
                vuids: &["VUID-vkCmdClearDepthStencilImage-image-00014"],
                ..Default::default()
            }));
        }

        if !matches!(
            image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(Box::new(ValidationError {
                context: "image_layout".into(),
                problem: "is not `ImageLayout::TransferDstOptimal` or `ImageLayout::General`"
                    .into(),
                vuids: &["VUID-vkCmdClearDepthStencilImage-imageLayout-00012"],
                ..Default::default()
            }));
        }

        if !(0.0..=1.0).contains(&clear_value.depth)
            && !device.enabled_extensions().ext_depth_range_unrestricted
        {
            return Err(Box::new(ValidationError {
                context: "clear_value.depth".into(),
                problem: "is not between `0.0` and `1.0` inclusive".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_depth_range_unrestricted",
                )])]),
                vuids: &["VUID-VkClearDepthStencilValue-depth-00022"],
            }));
        }

        for (region_index, subresource_range) in regions.iter().enumerate() {
            subresource_range
                .validate(device)
                .map_err(|err| err.add_context(format!("regions[{}]", region_index)))?;

            if !image_aspects.contains(subresource_range.aspects) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].aspects` is not a subset of `image.format().aspects()`",
                        region_index
                    )
                    .into(),
                    vuids: &[
                        "VUID-vkCmdClearDepthStencilImage-aspectMask-02824",
                        "VUID-vkCmdClearDepthStencilImage-image-02825",
                        "VUID-vkCmdClearDepthStencilImage-image-02826",
                    ],
                    ..Default::default()
                }));
            }

            if subresource_range.aspects.intersects(ImageAspects::STENCIL)
                && !image
                    .stencil_usage()
                    .unwrap_or(image.usage())
                    .intersects(ImageUsage::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].aspects` contains `ImageAspects::STENCIL`, but \
                        `image.stencil_usage()` does not contain `ImageUsage::TRANSFER_DST`",
                        region_index
                    )
                    .into(),
                    vuids: &[
                        "VUID-vkCmdClearDepthStencilImage-pRanges-02658",
                        "VUID-vkCmdClearDepthStencilImage-pRanges-02659",
                    ],
                    ..Default::default()
                }));
            }

            if !(subresource_range.aspects - ImageAspects::STENCIL).is_empty()
                && !image.usage().intersects(ImageUsage::TRANSFER_DST)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].aspects` contains aspects other than \
                        `ImageAspects::STENCIL`, but \
                        `image.usage()` does not contain `ImageUsage::TRANSFER_DST`",
                        region_index
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearDepthStencilImage-pRanges-02660"],
                    ..Default::default()
                }));
            }

            if subresource_range.mip_levels.end > image.mip_levels() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].mip_levels.end` is greater than `image.mip_levels()`",
                        region_index
                    )
                    .into(),
                    vuids: &[
                        "VUID-vkCmdClearDepthStencilImage-baseMipLevel-01474",
                        "VUID-vkCmdClearDepthStencilImage-pRanges-01694",
                    ],
                    ..Default::default()
                }));
            }

            if subresource_range.array_layers.end > image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`regions[{}].array_layers.end` is greater than `image.array_layers()`",
                        region_index
                    )
                    .into(),
                    vuids: &[
                        "VUID-vkCmdClearDepthStencilImage-baseArrayLayer-01476",
                        "VUID-vkCmdClearDepthStencilImage-pRanges-01695",
                    ],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> ClearDepthStencilImageInfoVk {
        let &Self {
            ref image,
            image_layout,
            clear_value,
            regions: _,
            _ne: _,
        } = self;

        ClearDepthStencilImageInfoVk {
            image: image.handle(),
            image_layout: image_layout.into(),
            depth_stencil: clear_value.to_vk(),
        }
    }

    pub(crate) fn to_vk_ranges(&self) -> SmallVec<[vk::ImageSubresourceRange; 8]> {
        self.regions
            .iter()
            .map(ImageSubresourceRange::to_vk)
            .collect()
    }
}

pub(crate) struct ClearDepthStencilImageInfoVk {
    pub(crate) image: vk::Image,
    pub(crate) image_layout: vk::ImageLayout,
    pub(crate) depth_stencil: vk::ClearDepthStencilValue,
}
