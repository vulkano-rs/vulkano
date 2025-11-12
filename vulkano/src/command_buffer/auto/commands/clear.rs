use crate::{
    buffer::{BufferContents, Subbuffer},
    command_buffer::{
        auto::Resource, sys::RecordingCommandBuffer, AutoCommandBufferBuilder, ClearColorImageInfo,
        ClearDepthStencilImageInfo, ResourceInCommand,
    },
    sync::PipelineStageAccessFlags,
    DeviceSize, SafeDeref, ValidationError,
};

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
