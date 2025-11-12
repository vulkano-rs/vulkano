use crate::{
    command_buffer::{
        auto::{
            BeginRenderPassState, BeginRenderingState, RenderPassState, RenderPassStateAttachments,
            RenderPassStateType, Resource,
        },
        sys::RecordingCommandBuffer,
        AutoCommandBufferBuilder, ClearAttachment, ClearRect, RenderPassBeginInfo,
        RenderingAttachmentInfo, RenderingAttachmentResolveInfo, RenderingInfo, ResourceInCommand,
        SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    pipeline::graphics::subpass::OwnedPipelineRenderingCreateInfo,
    sync::PipelineStageAccessFlags,
    ValidationError,
};
use smallvec::SmallVec;
use std::cmp::min;

/// # Commands for render passes.
///
/// These commands require a graphics queue.
impl<L> AutoCommandBufferBuilder<L> {
    /// Begins a render pass using a render pass object and framebuffer.
    ///
    /// You must call this or `begin_rendering` before you can record draw commands.
    ///
    /// `contents` specifies what kinds of commands will be recorded in the render pass, either
    /// draw commands or executions of secondary command buffers.
    pub fn begin_render_pass(
        &mut self,
        render_pass_begin_info: RenderPassBeginInfo,
        subpass_begin_info: SubpassBeginInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_begin_render_pass(&render_pass_begin_info, &subpass_begin_info)?;

        Ok(unsafe { self.begin_render_pass_unchecked(render_pass_begin_info, subpass_begin_info) })
    }

    fn validate_begin_render_pass(
        &self,
        render_pass_begin_info: &RenderPassBeginInfo,
        subpass_begin_info: &SubpassBeginInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_begin_render_pass(render_pass_begin_info, subpass_begin_info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is already active".into(),
                vuids: &["VUID-vkCmdBeginRenderPass2-renderpass"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBeginRenderPass2-initialLayout-03100
        // TODO:

        // VUID-vkCmdBeginRenderPass2-srcStageMask-06453
        // TODO:

        // VUID-vkCmdBeginRenderPass2-dstStageMask-06454
        // TODO:

        // VUID-vkCmdBeginRenderPass2-framebuffer-02533
        // For any attachment in framebuffer that is used by renderPass and is bound to memory
        // locations that are also bound to another attachment used by renderPass, and if at least
        // one of those uses causes either attachment to be written to, both attachments
        // must have had the VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT set

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_render_pass_unchecked(
        &mut self,
        render_pass_begin_info: RenderPassBeginInfo,
        subpass_begin_info: SubpassBeginInfo,
    ) -> &mut Self {
        let &RenderPassBeginInfo {
            ref render_pass,
            ref framebuffer,
            render_area_offset,
            render_area_extent,
            clear_values: _,
            _ne: _,
        } = &render_pass_begin_info;

        let subpass = render_pass.first_subpass();
        self.builder_state.render_pass = Some(RenderPassState {
            contents: subpass_begin_info.contents,
            render_area_offset,
            render_area_extent,

            rendering_info: OwnedPipelineRenderingCreateInfo::from_subpass(&subpass),
            attachments: Some(RenderPassStateAttachments::from_subpass(
                &subpass,
                framebuffer,
            )),

            render_pass: BeginRenderPassState {
                subpass,
                framebuffer: Some(framebuffer.clone()),
            }
            .into(),
        });

        self.add_render_pass_begin(
            "begin_render_pass",
            render_pass
                .attachments()
                .iter()
                .enumerate()
                .map(|(index, desc)| {
                    let image_view = &framebuffer.attachments()[index];
                    let index = index as u32;

                    (
                        ResourceInCommand::FramebufferAttachment { index }.into(),
                        Resource::Image {
                            image: image_view.image().clone(),
                            subresource_range: *image_view.subresource_range(),
                            // TODO: suboptimal
                            memory_access: PipelineStageAccessFlags::FragmentShader_InputAttachmentRead
                                | PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentRead
                                | PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentWrite
                                | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite,
                            start_layout: desc.initial_layout,
                            end_layout: desc.final_layout,
                        },
                    )
                })
                .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.begin_render_pass_unchecked(&render_pass_begin_info, &subpass_begin_info) };
            },
        );

        self
    }

    /// Advances to the next subpass of the render pass previously begun with `begin_render_pass`.
    pub fn next_subpass(
        &mut self,
        subpass_end_info: SubpassEndInfo,
        subpass_begin_info: SubpassBeginInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_next_subpass(&subpass_end_info, &subpass_begin_info)?;

        Ok(unsafe { self.next_subpass_unchecked(subpass_end_info, subpass_begin_info) })
    }

    fn validate_next_subpass(
        &self,
        subpass_end_info: &SubpassEndInfo,
        subpass_begin_info: &SubpassBeginInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_next_subpass(subpass_end_info, subpass_begin_info)?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdNextSubpass2-renderpass"],
                ..Default::default()
            })
        })?;

        let begin_render_pass_state = match &render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(state) => state,
            RenderPassStateType::BeginRendering(_) => {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance was not begun with \
                        `begin_render_pass`"
                        .into(),
                    // vuids?
                    ..Default::default()
                }));
            }
        };

        if begin_render_pass_state.subpass.is_last_subpass() {
            return Err(Box::new(ValidationError {
                problem: "the current subpass is the last subpass of the render pass".into(),
                vuids: &["VUID-vkCmdNextSubpass2-None-03102"],
                ..Default::default()
            }));
        }

        if self
            .builder_state
            .queries
            .values()
            .any(|state| state.in_subpass)
        {
            return Err(Box::new(ValidationError {
                problem: "a query that was begun in the current subpass is still active".into(),
                // vuids?
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn next_subpass_unchecked(
        &mut self,
        subpass_end_info: SubpassEndInfo,
        subpass_begin_info: SubpassBeginInfo,
    ) -> &mut Self {
        let render_pass_state = self.builder_state.render_pass.as_mut().unwrap();
        let begin_render_pass_state = match &mut render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(x) => x,
            _ => unreachable!(),
        };

        begin_render_pass_state.subpass.next_subpass();
        render_pass_state.contents = subpass_begin_info.contents;
        render_pass_state.rendering_info =
            OwnedPipelineRenderingCreateInfo::from_subpass(&begin_render_pass_state.subpass);
        render_pass_state.attachments = Some(RenderPassStateAttachments::from_subpass(
            &begin_render_pass_state.subpass,
            begin_render_pass_state.framebuffer.as_ref().unwrap(),
        ));

        if render_pass_state.rendering_info.as_ref().view_mask != 0 {
            // When multiview is enabled, at the beginning of each subpass, all
            // non-render pass state is undefined.
            self.builder_state.reset_non_render_pass_states();
        }

        self.add_command(
            "next_subpass",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.next_subpass_unchecked(&subpass_end_info, &subpass_begin_info) };
            },
        );

        self
    }

    /// Ends the render pass previously begun with `begin_render_pass`.
    ///
    /// This must be called after you went through all the subpasses.
    pub fn end_render_pass(
        &mut self,
        subpass_end_info: SubpassEndInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_render_pass(&subpass_end_info)?;

        Ok(unsafe { self.end_render_pass_unchecked(subpass_end_info) })
    }

    fn validate_end_render_pass(
        &self,
        subpass_end_info: &SubpassEndInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_end_render_pass(subpass_end_info)?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdEndRenderPass2-renderpass"],
                ..Default::default()
            })
        })?;

        let begin_render_pass_state = match &render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(state) => state,
            RenderPassStateType::BeginRendering(_) => {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance was not begun with \
                        `begin_render_pass`"
                        .into(),
                    vuids: &["VUID-vkCmdEndRenderPass2-None-06171"],
                    ..Default::default()
                }));
            }
        };

        if !begin_render_pass_state.subpass.is_last_subpass() {
            return Err(Box::new(ValidationError {
                problem: "the current subpass is not the last subpass of the render pass".into(),
                vuids: &["VUID-vkCmdEndRenderPass2-None-03103"],
                ..Default::default()
            }));
        }

        if self
            .builder_state
            .queries
            .values()
            .any(|state| state.in_subpass)
        {
            return Err(Box::new(ValidationError {
                problem: "a query that was begun in the current subpass is still active".into(),
                vuids: &["VUID-vkCmdEndRenderPass2-None-07005"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_render_pass_unchecked(
        &mut self,
        subpass_end_info: SubpassEndInfo,
    ) -> &mut Self {
        self.builder_state.render_pass = None;

        self.add_render_pass_end(
            "end_render_pass",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.end_render_pass_unchecked(&subpass_end_info) };
            },
        );

        self
    }

    /// Begins a render pass without a render pass object or framebuffer.
    ///
    /// Requires the [`dynamic_rendering`] device feature.
    ///
    /// You must call this or `begin_render_pass` before you can record draw commands.
    ///
    /// [`dynamic_rendering`]: crate::device::DeviceFeatures::dynamic_rendering
    pub fn begin_rendering(
        &mut self,
        mut rendering_info: RenderingInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        rendering_info.set_auto_extent_layers();
        self.validate_begin_rendering(&rendering_info)?;

        Ok(unsafe { self.begin_rendering_unchecked(rendering_info) })
    }

    fn validate_begin_rendering(
        &self,
        rendering_info: &RenderingInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_begin_rendering(rendering_info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is already active".into(),
                vuids: &["VUID-vkCmdBeginRendering-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_rendering_unchecked(
        &mut self,
        mut rendering_info: RenderingInfo,
    ) -> &mut Self {
        rendering_info.set_auto_extent_layers();

        let &RenderingInfo {
            render_area_offset,
            render_area_extent,
            layer_count: _,
            view_mask: _,
            ref color_attachments,
            ref depth_attachment,
            ref stencil_attachment,
            contents,
            _ne,
        } = &rendering_info;

        self.builder_state.render_pass = Some(RenderPassState {
            contents,
            render_area_offset,
            render_area_extent,

            rendering_info: OwnedPipelineRenderingCreateInfo::from_rendering_info(&rendering_info),
            attachments: Some(RenderPassStateAttachments::from_rendering_info(
                &rendering_info,
            )),

            render_pass: BeginRenderingState {
                pipeline_used: false,
            }
            .into(),
        });

        self.add_render_pass_begin(
            "begin_rendering",
            (color_attachments
                .iter()
                .enumerate()
                .filter_map(|(index, attachment_info)| {
                    attachment_info
                        .as_ref()
                        .map(|attachment_info| (index as u32, attachment_info))
                })
                .flat_map(|(index, attachment_info)| {
                    let &RenderingAttachmentInfo {
                        ref image_view,
                        image_layout,
                        ref resolve_info,
                        load_op: _,
                        store_op: _,
                        clear_value: _,
                        _ne: _,
                    } = attachment_info;

                    [
                        Some((
                            ResourceInCommand::ColorAttachment { index }.into(),
                            Resource::Image {
                                image: image_view.image().clone(),
                                subresource_range: *image_view.subresource_range(),
                                // TODO: suboptimal
                                memory_access: PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentRead
                                    | PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentWrite,
                                start_layout: image_layout,
                                end_layout: image_layout,
                            },
                        )),
                        resolve_info.as_ref().map(|resolve_info| {
                            let &RenderingAttachmentResolveInfo {
                                mode: _,
                                ref image_view,
                                image_layout,
                            } = resolve_info;

                            (
                                ResourceInCommand::ColorResolveAttachment { index }.into(),
                                Resource::Image {
                                    image: image_view.image().clone(),
                                    subresource_range: *image_view.subresource_range(),
                                    // TODO: suboptimal
                                    memory_access: PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentRead
                                        | PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentWrite,
                                    start_layout: image_layout,
                                    end_layout: image_layout,
                                },
                            )
                        }),
                    ]
                    .into_iter()
                    .flatten()
                }))
            .chain(depth_attachment.iter().flat_map(|attachment_info| {
                let &RenderingAttachmentInfo {
                    ref image_view,
                    image_layout,
                    ref resolve_info,
                    load_op: _,
                    store_op: _,
                    clear_value: _,
                    _ne: _,
                } = attachment_info;

                [
                    Some((
                        ResourceInCommand::DepthStencilAttachment.into(),
                        Resource::Image {
                            image: image_view.image().clone(),
                            subresource_range: *image_view.subresource_range(),
                            // TODO: suboptimal
                            memory_access: PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite,
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )),
                    resolve_info.as_ref().map(|resolve_info| {
                        let &RenderingAttachmentResolveInfo {
                            mode: _,
                            ref image_view,
                            image_layout,
                        } = resolve_info;

                        (
                            ResourceInCommand::DepthStencilResolveAttachment.into(),
                            Resource::Image {
                                image: image_view.image().clone(),
                                subresource_range: *image_view.subresource_range(),
                                // TODO: suboptimal
                                memory_access: PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentRead
                                    | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                                    | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentRead
                                    | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite,
                                start_layout: image_layout,
                                end_layout: image_layout,
                            },
                        )
                    }),
                ]
                .into_iter()
                .flatten()
            }))
            .chain(stencil_attachment.iter().flat_map(|attachment_info| {
                let &RenderingAttachmentInfo {
                    ref image_view,
                    image_layout,
                    ref resolve_info,
                    load_op: _,
                    store_op: _,
                    clear_value: _,
                    _ne: _,
                } = attachment_info;

                [
                    Some((
                        ResourceInCommand::DepthStencilAttachment.into(),
                        Resource::Image {
                            image: image_view.image().clone(),
                            subresource_range: *image_view.subresource_range(),
                            // TODO: suboptimal
                            memory_access: PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentRead
                                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite,
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )),
                    resolve_info.as_ref().map(|resolve_info| {
                        let &RenderingAttachmentResolveInfo {
                            mode: _,
                            ref image_view,
                            image_layout,
                        } = resolve_info;

                        (
                            ResourceInCommand::DepthStencilResolveAttachment.into(),
                            Resource::Image {
                                image: image_view.image().clone(),
                                subresource_range: *image_view.subresource_range(),
                                // TODO: suboptimal
                                memory_access: PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentRead
                                    | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                                    | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentRead
                                    | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite,
                                start_layout: image_layout,
                                end_layout: image_layout,
                            },
                        )
                    }),
                ]
                .into_iter()
                .flatten()
            }))
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.begin_rendering_unchecked(&rendering_info) };
            },
        );

        self
    }

    /// Ends the render pass previously begun with `begin_rendering`.
    pub fn end_rendering(&mut self) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_rendering()?;

        Ok(unsafe { self.end_rendering_unchecked() })
    }

    fn validate_end_rendering(&self) -> Result<(), Box<ValidationError>> {
        self.inner.validate_end_rendering()?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &[
                    "VUID-vkCmdEndRendering-renderpass",
                    "VUID-vkCmdEndRendering-commandBuffer-06162",
                ],
                ..Default::default()
            })
        })?;

        match &render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(_) => {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance was not begun with \
                        `begin_rendering`"
                        .into(),
                    vuids: &["VUID-vkCmdEndRendering-None-06161"],
                    ..Default::default()
                }))
            }
            RenderPassStateType::BeginRendering(_) => (),
        }

        if self
            .builder_state
            .queries
            .values()
            .any(|state| state.in_subpass)
        {
            return Err(Box::new(ValidationError {
                problem: "a query that was begun in the current render pass instance \
                    is still active"
                    .into(),
                vuids: &["VUID-vkCmdEndRendering-None-06999"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_rendering_unchecked(&mut self) -> &mut Self {
        self.builder_state.render_pass = None;

        self.add_render_pass_end(
            "end_rendering",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.end_rendering_unchecked() };
            },
        );

        self
    }

    /// Clears specific regions of specific attachments of the framebuffer.
    ///
    /// `attachments` specify the types of attachments and their clear values.
    /// `rects` specify the regions to clear.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). And the command must be inside
    /// render pass.
    ///
    /// If the render pass instance this is recorded in uses multiview,
    /// then `ClearRect.base_array_layer` must be zero and `ClearRect.layer_count` must be one.
    ///
    /// The rectangle area must be inside the render area ranges.
    pub fn clear_attachments(
        &mut self,
        attachments: SmallVec<[ClearAttachment; 4]>,
        rects: SmallVec<[ClearRect; 4]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_clear_attachments(&attachments, &rects)?;

        Ok(unsafe { self.clear_attachments_unchecked(attachments, rects) })
    }

    fn validate_clear_attachments(
        &self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_clear_attachments(attachments, rects)?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdClearAttachments-renderpass"],
                ..Default::default()
            })
        })?;

        if render_pass_state.contents != SubpassContents::Inline {
            return Err(Box::new(ValidationError {
                problem: "the contents of the current subpass instance is not \
                    `SubpassContents::Inline`"
                    .into(),
                // vuids?
                ..Default::default()
            }));
        }

        let mut layer_count = u32::MAX;

        for (clear_index, &clear_attachment) in attachments.iter().enumerate() {
            match clear_attachment {
                ClearAttachment::Color {
                    color_attachment,
                    clear_value,
                } => {
                    let attachment_format = *render_pass_state
                        .rendering_info
                        .as_ref()
                        .color_attachment_formats
                        .get(color_attachment as usize)
                        .ok_or_else(|| {
                            Box::new(ValidationError {
                                context: format!("attachments[{}].color_attachment", clear_index)
                                    .into(),
                                problem: "is not less than the number of color attachments in the \
                                current subpass instance"
                                    .into(),
                                vuids: &["VUID-vkCmdClearAttachments-aspectMask-07271"],
                                ..Default::default()
                            })
                        })?;

                    if let Some(attachment_format) = attachment_format {
                        let required_numeric_type = attachment_format
                            .numeric_format_color()
                            .unwrap()
                            .numeric_type();

                        if clear_value.numeric_type() != required_numeric_type {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`attachments[{0}].clear_value` is `ClearColorValue::{1:?}`, \
                                    but the color attachment specified by \
                                    `attachments[{0}].color_attachment` requires a clear value \
                                    of type `ClearColorValue::{2:?}`",
                                    clear_index,
                                    clear_value.numeric_type(),
                                    required_numeric_type,
                                )
                                .into(),
                                vuids: &["VUID-vkCmdClearAttachments-aspectMask-02501"],
                                ..Default::default()
                            }));
                        }
                    }

                    let image_view = render_pass_state
                        .attachments
                        .as_ref()
                        .and_then(|attachments| attachments.depth_attachment.as_ref())
                        .map(|attachment_info| &attachment_info.image_view);

                    // We only know the layer count if we have a known attachment image.
                    if let Some(image_view) = image_view {
                        layer_count = min(layer_count, image_view.subresource_range().layer_count);
                    }
                }
                ClearAttachment::Depth(_)
                | ClearAttachment::Stencil(_)
                | ClearAttachment::DepthStencil(_) => {
                    if matches!(
                        clear_attachment,
                        ClearAttachment::Depth(_) | ClearAttachment::DepthStencil(_)
                    ) && render_pass_state
                        .rendering_info
                        .as_ref()
                        .depth_attachment_format
                        .is_none()
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`attachments[{0}]` is `ClearAttachment::Depth` or \
                                `ClearAttachment::DepthStencil`, but \
                                the current subpass instance does not have a depth attachment",
                                clear_index,
                            )
                            .into(),
                            vuids: &["VUID-vkCmdClearAttachments-aspectMask-02502"],
                            ..Default::default()
                        }));
                    }

                    if matches!(
                        clear_attachment,
                        ClearAttachment::Stencil(_) | ClearAttachment::DepthStencil(_)
                    ) && render_pass_state
                        .rendering_info
                        .as_ref()
                        .stencil_attachment_format
                        .is_none()
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`attachments[{0}]` is `ClearAttachment::Stencil` or \
                                `ClearAttachment::DepthStencil`, but \
                                the current subpass instance does not have a stencil attachment",
                                clear_index,
                            )
                            .into(),
                            vuids: &["VUID-vkCmdClearAttachments-aspectMask-02503"],
                            ..Default::default()
                        }));
                    }

                    let image_view = render_pass_state
                        .attachments
                        .as_ref()
                        .and_then(|attachments| attachments.depth_attachment.as_ref())
                        .map(|attachment_info| &attachment_info.image_view);

                    // We only know the layer count if we have a known attachment image.
                    if let Some(image_view) = image_view {
                        layer_count = min(layer_count, image_view.subresource_range().layer_count);
                    }
                }
            }
        }

        for (rect_index, rect) in rects.iter().enumerate() {
            for i in 0..2 {
                // TODO: This check will always pass in secondary command buffers because of how
                // it's set in `with_level`.
                // It needs to be checked during `execute_commands` instead.

                if rect.offset[i] < render_pass_state.render_area_offset[i] {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`rects[{0}].offset[{i}]` is less than \
                            `render_area_offset[{i}]` of the current render pass instance",
                            rect_index
                        )
                        .into(),
                        vuids: &["VUID-vkCmdClearAttachments-pRects-00016"],
                        ..Default::default()
                    }));
                }

                if rect.offset[i] + rect.extent[i]
                    > render_pass_state.render_area_offset[i]
                        + render_pass_state.render_area_extent[i]
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`rects[{0}].offset[{i}] + rects[{0}].extent[{i}]` is \
                            greater than `render_area_offset[{i}] + render_area_extent[{i}]` \
                            of the current render pass instance",
                            rect_index,
                        )
                        .into(),
                        vuids: &["VUID-vkCmdClearAttachments-pRects-00016"],
                        ..Default::default()
                    }));
                }
            }

            if !rect
                .base_array_layer
                .checked_add(rect.layer_count)
                .is_some_and(|end| end <= layer_count)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`rects[{0}].base_array_layer + rects[{0}].layer_count` is greater than \
                        the number of array layers in the current render pass instance",
                        rect_index,
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearAttachments-pRects-06937"],
                    ..Default::default()
                }));
            }

            if render_pass_state.rendering_info.as_ref().view_mask != 0
                && !(rect.base_array_layer == 0 && rect.layer_count == 1)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "the current render pass instance has a non-zero `view_mask`, but \
                        `(rects[{0}].base_array_layer, rects[{0}].layer_count)` is not `(0, 1)`",
                        rect_index,
                    )
                    .into(),
                    vuids: &["VUID-vkCmdClearAttachments-baseArrayLayer-00018"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_attachments_unchecked(
        &mut self,
        attachments: SmallVec<[ClearAttachment; 4]>,
        rects: SmallVec<[ClearRect; 4]>,
    ) -> &mut Self {
        self.add_command(
            "clear_attachments",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.clear_attachments_unchecked(&attachments, &rects) };
            },
        );

        self
    }
}
