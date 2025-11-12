use crate::{
    command_buffer::{
        auto::Resource, sys::RecordingCommandBuffer, AutoCommandBufferBuilder, BlitImageInfo,
        BufferCopy, BufferImageCopy, CopyBufferInfo, CopyBufferToImageInfo, CopyImageInfo,
        CopyImageToBufferInfo, ImageBlit, ImageCopy, ImageResolve, ResolveImageInfo,
        ResourceInCommand,
    },
    sync::PipelineStageAccessFlags,
    ValidationError,
};

/// # Commands to transfer data between resources.
impl<L> AutoCommandBufferBuilder<L> {
    /// Copies data from a buffer to another buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `src_buffer` or `dst_buffer` were not created from the same device as `self`.
    pub fn copy_buffer(
        &mut self,
        copy_buffer_info: impl Into<CopyBufferInfo>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let copy_buffer_info = copy_buffer_info.into();
        self.validate_copy_buffer(&copy_buffer_info)?;

        Ok(unsafe { self.copy_buffer_unchecked(copy_buffer_info) })
    }

    fn validate_copy_buffer(
        &self,
        copy_buffer_info: &CopyBufferInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_copy_buffer(copy_buffer_info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyBuffer2-renderpass"],
                ..Default::default()
            }));
        }

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
        } = &copy_buffer_info;

        self.add_command(
            "copy_buffer",
            regions
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
                            ResourceInCommand::Source.into(),
                            Resource::Buffer {
                                buffer: src_buffer.clone(),
                                range: src_offset..src_offset + size,
                                memory_access: PipelineStageAccessFlags::Copy_TransferRead,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Buffer {
                                buffer: dst_buffer.clone(),
                                range: dst_offset..dst_offset + size,
                                memory_access: PipelineStageAccessFlags::Copy_TransferWrite,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.copy_buffer_unchecked(&copy_buffer_info) };
            },
        );

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
    /// - Panics if `src_image` or `dst_image` were not created from the same device as `self`.
    pub fn copy_image(
        &mut self,
        copy_image_info: CopyImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_image(&copy_image_info)?;

        Ok(unsafe { self.copy_image_unchecked(copy_image_info) })
    }

    fn validate_copy_image(
        &self,
        copy_image_info: &CopyImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_copy_image(copy_image_info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyImage2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_image_unchecked(&mut self, copy_image_info: CopyImageInfo) -> &mut Self {
        let &CopyImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = &copy_image_info;

        self.add_command(
            "copy_image",
            regions
                .iter()
                .flat_map(|region| {
                    let &ImageCopy {
                        src_subresource,
                        src_offset: _,
                        dst_subresource,
                        dst_offset: _,
                        extent: _,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Image {
                                image: src_image.clone(),
                                subresource_range: src_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Copy_TransferRead,
                                start_layout: src_image_layout,
                                end_layout: src_image_layout,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Image {
                                image: dst_image.clone(),
                                subresource_range: dst_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Copy_TransferWrite,
                                start_layout: dst_image_layout,
                                end_layout: dst_image_layout,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.copy_image_unchecked(&copy_image_info) };
            },
        );

        self
    }

    /// Copies from a buffer to an image.
    pub fn copy_buffer_to_image(
        &mut self,
        copy_buffer_to_image_info: CopyBufferToImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_buffer_to_image(&copy_buffer_to_image_info)?;

        Ok(unsafe { self.copy_buffer_to_image_unchecked(copy_buffer_to_image_info) })
    }

    fn validate_copy_buffer_to_image(
        &self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_copy_buffer_to_image(copy_buffer_to_image_info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyBufferToImage2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_buffer_to_image_unchecked(
        &mut self,
        copy_buffer_to_image_info: CopyBufferToImageInfo,
    ) -> &mut Self {
        let &CopyBufferToImageInfo {
            ref src_buffer,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = &copy_buffer_to_image_info;

        self.add_command(
            "copy_buffer_to_image",
            regions
                .iter()
                .flat_map(|region| {
                    let &BufferImageCopy {
                        buffer_offset,
                        buffer_row_length: _,
                        buffer_image_height: _,
                        image_subresource,
                        image_offset: _,
                        image_extent: _,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Buffer {
                                buffer: src_buffer.clone(),
                                range: buffer_offset
                                    ..buffer_offset + region.buffer_copy_size(dst_image.format()),
                                memory_access: PipelineStageAccessFlags::Copy_TransferRead,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Image {
                                image: dst_image.clone(),
                                subresource_range: image_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Copy_TransferWrite,
                                start_layout: dst_image_layout,
                                end_layout: dst_image_layout,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.copy_buffer_to_image_unchecked(&copy_buffer_to_image_info) };
            },
        );

        self
    }

    /// Copies from an image to a buffer.
    pub fn copy_image_to_buffer(
        &mut self,
        copy_image_to_buffer_info: CopyImageToBufferInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_image_to_buffer(&copy_image_to_buffer_info)?;

        Ok(unsafe { self.copy_image_to_buffer_unchecked(copy_image_to_buffer_info) })
    }

    fn validate_copy_image_to_buffer(
        &self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_copy_image_to_buffer(copy_image_to_buffer_info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyImageToBuffer2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_image_to_buffer_unchecked(
        &mut self,
        copy_image_to_buffer_info: CopyImageToBufferInfo,
    ) -> &mut Self {
        let &CopyImageToBufferInfo {
            ref src_image,
            src_image_layout,
            ref dst_buffer,
            ref regions,
            _ne: _,
        } = &copy_image_to_buffer_info;

        self.add_command(
            "copy_image_to_buffer",
            regions
                .iter()
                .flat_map(|region| {
                    let &BufferImageCopy {
                        buffer_offset,
                        buffer_row_length: _,
                        buffer_image_height: _,
                        image_subresource,
                        image_offset: _,
                        image_extent: _,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Image {
                                image: src_image.clone(),
                                subresource_range: image_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Copy_TransferRead,
                                start_layout: src_image_layout,
                                end_layout: src_image_layout,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Buffer {
                                buffer: dst_buffer.clone(),
                                range: buffer_offset
                                    ..buffer_offset + region.buffer_copy_size(src_image.format()),
                                memory_access: PipelineStageAccessFlags::Copy_TransferWrite,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.copy_image_to_buffer_unchecked(&copy_image_to_buffer_info) };
            },
        );

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
    /// # Panics
    ///
    /// - Panics if the source or the destination was not created with `device`.
    pub fn blit_image(
        &mut self,
        blit_image_info: BlitImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_blit_image(&blit_image_info)?;

        Ok(unsafe { self.blit_image_unchecked(blit_image_info) })
    }

    fn validate_blit_image(
        &self,
        blit_image_info: &BlitImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_blit_image(blit_image_info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdBlitImage2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn blit_image_unchecked(&mut self, blit_image_info: BlitImageInfo) -> &mut Self {
        let &BlitImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            filter: _,
            _ne: _,
        } = &blit_image_info;

        self.add_command(
            "blit_image",
            regions
                .iter()
                .flat_map(|region| {
                    let &ImageBlit {
                        src_subresource,
                        src_offsets: _,
                        dst_subresource,
                        dst_offsets: _,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Image {
                                image: src_image.clone(),
                                subresource_range: src_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Blit_TransferRead,
                                start_layout: src_image_layout,
                                end_layout: src_image_layout,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Image {
                                image: dst_image.clone(),
                                subresource_range: dst_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Blit_TransferWrite,
                                start_layout: dst_image_layout,
                                end_layout: dst_image_layout,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.blit_image_unchecked(&blit_image_info) };
            },
        );

        self
    }

    /// Resolves a multisampled image into a single-sampled image.
    ///
    /// # Panics
    ///
    /// - Panics if `src_image` or `dst_image` were not created from the same device as `self`.
    pub fn resolve_image(
        &mut self,
        resolve_image_info: ResolveImageInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_resolve_image(&resolve_image_info)?;

        Ok(unsafe { self.resolve_image_unchecked(resolve_image_info) })
    }

    fn validate_resolve_image(
        &self,
        resolve_image_info: &ResolveImageInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_resolve_image(resolve_image_info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdResolveImage2-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn resolve_image_unchecked(
        &mut self,
        resolve_image_info: ResolveImageInfo,
    ) -> &mut Self {
        let &ResolveImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = &resolve_image_info;

        self.add_command(
            "resolve_image",
            regions
                .iter()
                .flat_map(|region| {
                    let &ImageResolve {
                        src_subresource,
                        src_offset: _,
                        dst_subresource,
                        dst_offset: _,
                        extent: _,
                        _ne: _,
                    } = region;

                    [
                        (
                            ResourceInCommand::Source.into(),
                            Resource::Image {
                                image: src_image.clone(),
                                subresource_range: src_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Resolve_TransferRead,
                                start_layout: src_image_layout,
                                end_layout: src_image_layout,
                            },
                        ),
                        (
                            ResourceInCommand::Destination.into(),
                            Resource::Image {
                                image: dst_image.clone(),
                                subresource_range: dst_subresource.into(),
                                memory_access: PipelineStageAccessFlags::Resolve_TransferWrite,
                                start_layout: dst_image_layout,
                                end_layout: dst_image_layout,
                            },
                        ),
                    ]
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.resolve_image_unchecked(&resolve_image_info) };
            },
        );

        self
    }
}
