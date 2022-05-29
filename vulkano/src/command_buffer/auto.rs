// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    commands::{
        pipeline::{
            CheckDescriptorSetsValidityError, CheckDispatchError, CheckDynamicStateValidityError,
            CheckIndexBufferError, CheckIndirectBufferError, CheckPipelineError,
            CheckPushConstantsValidityError, CheckVertexBufferError,
        },
        query::{
            CheckBeginQueryError, CheckCopyQueryPoolResultsError, CheckEndQueryError,
            CheckResetQueryPoolError, CheckWriteTimestampError,
        },
    },
    pool::{
        standard::{StandardCommandPoolAlloc, StandardCommandPoolBuilder},
        CommandPool, CommandPoolAlloc, CommandPoolBuilderAlloc,
    },
    synced::{
        CommandBufferState, SyncCommandBuffer, SyncCommandBufferBuilder,
        SyncCommandBufferBuilderError,
    },
    sys::{CommandBufferBeginInfo, UnsafeCommandBuffer},
    CommandBufferExecError, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderPassInfo,
    CommandBufferInheritanceRenderPassType, CommandBufferLevel, CommandBufferUsage,
    PrimaryCommandBuffer, RenderingAttachmentInfo, SecondaryCommandBuffer, SubpassContents,
};
use crate::{
    buffer::{sys::UnsafeBuffer, BufferAccess},
    command_buffer::CommandBufferInheritanceRenderingInfo,
    device::{physical::QueueFamily, Device, DeviceOwned, Queue},
    image::{sys::UnsafeImage, ImageAccess, ImageLayout, ImageSubresourceRange},
    pipeline::{graphics::render_pass::PipelineRenderPassType, GraphicsPipeline},
    query::{QueryControlFlags, QueryType},
    render_pass::{Framebuffer, Subpass},
    sync::{AccessCheckError, AccessFlags, GpuFuture, PipelineMemoryAccess, PipelineStages},
    DeviceSize, OomError,
};
use std::{
    collections::HashMap,
    error, fmt,
    marker::PhantomData,
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// Note that command buffers allocated from the default command pool (`Arc<StandardCommandPool>`)
/// don't implement the `Send` and `Sync` traits. If you use this pool, then the
/// `AutoCommandBufferBuilder` will not implement `Send` and `Sync` either. Once a command buffer
/// is built, however, it *does* implement `Send` and `Sync`.
pub struct AutoCommandBufferBuilder<L, P = StandardCommandPoolBuilder> {
    pub(super) inner: SyncCommandBufferBuilder,
    pool_builder_alloc: P, // Safety: must be dropped after `inner`

    // The queue family that this command buffer is being created for.
    queue_family_id: u32,

    // The inheritance for secondary command buffers.
    // Must be `None` in a primary command buffer and `Some` in a secondary command buffer.
    pub(super) inheritance_info: Option<CommandBufferInheritanceInfo>,

    // Usage flags passed when creating the command buffer.
    pub(super) usage: CommandBufferUsage,

    // If we're inside a render pass, contains the render pass state.
    pub(super) render_pass_state: Option<RenderPassState>,

    // If any queries are active, this hashmap contains their state.
    pub(super) query_state: HashMap<ash::vk::QueryType, QueryState>,

    _data: PhantomData<L>,
}

// The state of the current render pass, specifying the pass, subpass index and its intended contents.
pub(super) struct RenderPassState {
    pub(super) contents: SubpassContents,
    pub(super) render_area_offset: [u32; 2],
    pub(super) render_area_extent: [u32; 2],
    pub(super) render_pass: RenderPassStateType,
}

pub(super) enum RenderPassStateType {
    BeginRenderPass(BeginRenderPassState),
    BeginRendering(BeginRenderingState),
    Inherited,
}

impl From<BeginRenderPassState> for RenderPassStateType {
    #[inline]
    fn from(val: BeginRenderPassState) -> Self {
        Self::BeginRenderPass(val)
    }
}

impl From<BeginRenderingState> for RenderPassStateType {
    #[inline]
    fn from(val: BeginRenderingState) -> Self {
        Self::BeginRendering(val)
    }
}

pub(super) struct BeginRenderPassState {
    pub(super) subpass: Subpass,
    pub(super) framebuffer: Arc<Framebuffer>,
}

pub(super) struct BeginRenderingState {
    pub(super) view_mask: u32,
    pub(super) color_attachments: Vec<Option<RenderingAttachmentInfo>>,
    pub(super) depth_attachment: Option<RenderingAttachmentInfo>,
    pub(super) stencil_attachment: Option<RenderingAttachmentInfo>,
}

// The state of an active query.
pub(super) struct QueryState {
    pub(super) query_pool: ash::vk::QueryPool,
    pub(super) query: u32,
    pub(super) ty: QueryType,
    pub(super) flags: QueryControlFlags,
    pub(super) in_subpass: bool,
}

impl AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, StandardCommandPoolBuilder> {
    /// Starts recording a primary command buffer.
    #[inline]
    pub fn primary(
        device: Arc<Device>,
        queue_family: QueueFamily,
        usage: CommandBufferUsage,
    ) -> Result<
        AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, StandardCommandPoolBuilder>,
        CommandBufferBeginError,
    > {
        unsafe {
            AutoCommandBufferBuilder::begin(
                device,
                queue_family,
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage,
                    inheritance_info: None,
                    _ne: crate::NonExhaustive(()),
                },
            )
        }
    }
}

impl AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, StandardCommandPoolBuilder> {
    /// Starts recording a secondary command buffer.
    #[inline]
    pub fn secondary(
        device: Arc<Device>,
        queue_family: QueueFamily,
        usage: CommandBufferUsage,
        inheritance_info: CommandBufferInheritanceInfo,
    ) -> Result<
        AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, StandardCommandPoolBuilder>,
        CommandBufferBeginError,
    > {
        unsafe {
            Ok(AutoCommandBufferBuilder::begin(
                device,
                queue_family,
                CommandBufferLevel::Secondary,
                CommandBufferBeginInfo {
                    usage,
                    inheritance_info: Some(inheritance_info),
                    _ne: crate::NonExhaustive(()),
                },
            )?)
        }
    }
}

impl<L> AutoCommandBufferBuilder<L, StandardCommandPoolBuilder> {
    // Actual constructor. Private.
    //
    // `begin_info.inheritance_info` must match `level`.
    unsafe fn begin(
        device: Arc<Device>,
        queue_family: QueueFamily,
        level: CommandBufferLevel,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<AutoCommandBufferBuilder<L, StandardCommandPoolBuilder>, CommandBufferBeginError>
    {
        Self::validate_begin(&device, &queue_family, level, &begin_info)?;

        let &CommandBufferBeginInfo {
            usage,
            ref inheritance_info,
            _ne: _,
        } = &begin_info;

        let inheritance_info = inheritance_info.clone();
        let mut render_pass_state = None;

        if let Some(inheritance_info) = &inheritance_info {
            let &CommandBufferInheritanceInfo {
                ref render_pass,
                occlusion_query,
                query_statistics_flags,
                _ne: _,
            } = inheritance_info;

            if let Some(render_pass) = render_pass {
                // In a secondary command buffer, we don't know the render area yet, so use a
                // dummy value.
                let render_area_offset = [0, 0];
                let mut render_area_extent = [u32::MAX, u32::MAX];

                if let CommandBufferInheritanceRenderPassType::BeginRenderPass(
                    CommandBufferInheritanceRenderPassInfo {
                        framebuffer: Some(framebuffer),
                        ..
                    },
                ) = render_pass
                {
                    // Still not exact, but it's a better upper bound.
                    render_area_extent = framebuffer.extent();
                }

                render_pass_state = Some(RenderPassState {
                    contents: SubpassContents::Inline,
                    render_area_offset,
                    render_area_extent,
                    render_pass: RenderPassStateType::Inherited,
                });
            }
        }

        let pool_builder_alloc = Device::standard_command_pool(&device, queue_family)
            .allocate(level, 1)?
            .next()
            .expect("Requested one command buffer from the command pool, but got zero.");
        let inner = SyncCommandBufferBuilder::new(pool_builder_alloc.inner(), begin_info)?;

        Ok(AutoCommandBufferBuilder {
            inner,
            pool_builder_alloc,
            queue_family_id: queue_family.id(),
            render_pass_state,
            query_state: HashMap::default(),
            inheritance_info,
            usage,
            _data: PhantomData,
        })
    }

    fn validate_begin(
        device: &Device,
        queue_family: &QueueFamily,
        level: CommandBufferLevel,
        begin_info: &CommandBufferBeginInfo,
    ) -> Result<(), CommandBufferBeginError> {
        let physical_device = device.physical_device();
        let properties = physical_device.properties();

        let &CommandBufferBeginInfo {
            usage,
            ref inheritance_info,
            _ne: _,
        } = &begin_info;

        if let Some(inheritance_info) = &inheritance_info {
            debug_assert!(level == CommandBufferLevel::Secondary);

            let &CommandBufferInheritanceInfo {
                ref render_pass,
                occlusion_query,
                query_statistics_flags,
                _ne: _,
            } = inheritance_info;

            if let Some(render_pass) = render_pass {
                // VUID-VkCommandBufferBeginInfo-flags-06000
                // VUID-VkCommandBufferBeginInfo-flags-06002
                // Ensured by the definition of the `CommandBufferInheritanceRenderPassType` enum.

                match render_pass {
                    CommandBufferInheritanceRenderPassType::BeginRenderPass(render_pass_info) => {
                        let &CommandBufferInheritanceRenderPassInfo {
                            ref subpass,
                            ref framebuffer,
                        } = render_pass_info;

                        // VUID-VkCommandBufferInheritanceInfo-commonparent
                        assert_eq!(device, subpass.render_pass().device().as_ref());

                        // VUID-VkCommandBufferBeginInfo-flags-06001
                        // Ensured by how the `Subpass` type is constructed.

                        if let Some(framebuffer) = framebuffer {
                            // VUID-VkCommandBufferInheritanceInfo-commonparent
                            assert_eq!(device, framebuffer.device().as_ref());

                            // VUID-VkCommandBufferBeginInfo-flags-00055
                            if !framebuffer
                                .render_pass()
                                .is_compatible_with(subpass.render_pass())
                            {
                                return Err(CommandBufferBeginError::FramebufferNotCompatible);
                            }
                        }
                    }
                    CommandBufferInheritanceRenderPassType::BeginRendering(rendering_info) => {
                        let &CommandBufferInheritanceRenderingInfo {
                            view_mask,
                            ref color_attachment_formats,
                            depth_attachment_format,
                            stencil_attachment_format,
                            rasterization_samples,
                        } = rendering_info;

                        // VUID-VkCommandBufferInheritanceRenderingInfo-multiview-06008
                        if view_mask != 0 && !device.enabled_features().multiview {
                            return Err(CommandBufferBeginError::FeatureNotEnabled {
                                feature: "multiview",
                                reason: "view_mask is not 0",
                            });
                        }

                        let view_count = u32::BITS - view_mask.leading_zeros();

                        // VUID-VkCommandBufferInheritanceRenderingInfo-viewMask-06009
                        if view_count > properties.max_multiview_view_count.unwrap_or(0) {
                            return Err(CommandBufferBeginError::MaxMultiviewViewCountExceeded {
                                view_count,
                                max: properties.max_multiview_view_count.unwrap_or(0),
                            });
                        }

                        for (attachment_index, format) in color_attachment_formats
                            .iter()
                            .enumerate()
                            .flat_map(|(i, f)| f.map(|f| (i, f)))
                        {
                            let attachment_index = attachment_index as u32;

                            // VUID-VkCommandBufferInheritanceRenderingInfo-pColorAttachmentFormats-06006
                            if !physical_device
                                .format_properties(format)
                                .potential_format_features()
                                .color_attachment
                            {
                                return Err(
                                    CommandBufferBeginError::ColorAttachmentFormatUsageNotSupported {
                                        attachment_index,
                                    },
                                );
                            }
                        }

                        if let Some(format) = depth_attachment_format {
                            // VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06540
                            if !format.aspects().depth {
                                return Err(
                                    CommandBufferBeginError::DepthAttachmentFormatUsageNotSupported,
                                );
                            }

                            // VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06007
                            if !physical_device
                                .format_properties(format)
                                .potential_format_features()
                                .depth_stencil_attachment
                            {
                                return Err(
                                    CommandBufferBeginError::DepthAttachmentFormatUsageNotSupported,
                                );
                            }
                        }

                        if let Some(format) = stencil_attachment_format {
                            // VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-06541
                            if !format.aspects().stencil {
                                return Err(
                                    CommandBufferBeginError::StencilAttachmentFormatUsageNotSupported,
                                );
                            }

                            // VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-06199
                            if !physical_device
                                .format_properties(format)
                                .potential_format_features()
                                .depth_stencil_attachment
                            {
                                return Err(
                                    CommandBufferBeginError::StencilAttachmentFormatUsageNotSupported,
                                );
                            }
                        }

                        if let (Some(depth_format), Some(stencil_format)) =
                            (depth_attachment_format, stencil_attachment_format)
                        {
                            // VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06200
                            if depth_format != stencil_format {
                                return Err(
                                    CommandBufferBeginError::DepthStencilAttachmentFormatMismatch,
                                );
                            }
                        }
                    }
                }
            }

            if let Some(control_flags) = occlusion_query {
                // VUID-VkCommandBufferInheritanceInfo-occlusionQueryEnable-00056
                // VUID-VkCommandBufferInheritanceInfo-queryFlags-02788
                if !device.enabled_features().inherited_queries {
                    return Err(CommandBufferBeginError::FeatureNotEnabled {
                        feature: "inherited_queries",
                        reason: "occlusion queries were enabled",
                    });
                }

                // VUID-vkBeginCommandBuffer-commandBuffer-00052
                if control_flags.precise && !device.enabled_features().occlusion_query_precise {
                    return Err(CommandBufferBeginError::FeatureNotEnabled {
                        feature: "occlusion_query_precise",
                        reason: "occlusion_query.precise was set",
                    });
                }
            }

            // VUID-VkCommandBufferInheritanceInfo-pipelineStatistics-00058
            if query_statistics_flags.count() > 0
                && !device.enabled_features().pipeline_statistics_query
            {
                return Err(CommandBufferBeginError::FeatureNotEnabled {
                    feature: "pipeline_statistics_query",
                    reason: "one or more statistics flags were enabled",
                });
            }
        } else {
            debug_assert!(level == CommandBufferLevel::Primary);

            // VUID-vkBeginCommandBuffer-commandBuffer-02840
            // Ensured by the definition of the `CommandBufferUsage` enum.
        }

        Ok(())
    }
}

/// Error that can happen when beginning recording of a command buffer.
#[derive(Clone, Copy, Debug)]
pub enum CommandBufferBeginError {
    /// Not enough memory.
    OomError(OomError),

    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
    },

    /// A color attachment has a format that does not support that usage.
    ColorAttachmentFormatUsageNotSupported { attachment_index: u32 },

    /// The depth attachment has a format that does not support that usage.
    DepthAttachmentFormatUsageNotSupported,

    /// The depth and stencil attachments have different formats.
    DepthStencilAttachmentFormatMismatch,

    /// The framebuffer is not compatible with the render pass.
    FramebufferNotCompatible,

    /// The `max_multiview_view_count` limit has been exceeded.
    MaxMultiviewViewCountExceeded { view_count: u32, max: u32 },

    /// The stencil attachment has a format that does not support that usage.
    StencilAttachmentFormatUsageNotSupported,
}

impl error::Error for CommandBufferBeginError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for CommandBufferBeginError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::OomError(_) => write!(f, "not enough memory available"),

            Self::FeatureNotEnabled { feature, reason } => {
                write!(f, "the feature {} must be enabled: {}", feature, reason)
            }

            Self::ColorAttachmentFormatUsageNotSupported { attachment_index } => write!(
                f,
                "color attachment {} has a format that does not support that usage",
                attachment_index,
            ),
            Self::DepthAttachmentFormatUsageNotSupported => write!(
                f,
                "the depth attachment has a format that does not support that usage",
            ),
            Self::DepthStencilAttachmentFormatMismatch => write!(
                f,
                "the depth and stencil attachments have different formats",
            ),
            Self::FramebufferNotCompatible => {
                write!(f, "the framebuffer is not compatible with the render pass",)
            }
            Self::MaxMultiviewViewCountExceeded { .. } => {
                write!(f, "the `max_multiview_view_count` limit has been exceeded",)
            }
            Self::StencilAttachmentFormatUsageNotSupported => write!(
                f,
                "the stencil attachment has a format that does not support that usage",
            ),
        }
    }
}

impl From<OomError> for CommandBufferBeginError {
    #[inline]
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl<P> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>
where
    P: CommandPoolBuilderAlloc,
{
    /// Builds the command buffer.
    #[inline]
    pub fn build(self) -> Result<PrimaryAutoCommandBuffer<P::Alloc>, BuildError> {
        if self.render_pass_state.is_some() {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass.into());
        }

        if !self.query_state.is_empty() {
            return Err(AutoCommandBufferBuilderContextError::QueryIsActive.into());
        }

        let submit_state = match self.usage {
            CommandBufferUsage::MultipleSubmit => SubmitState::ExclusiveUse {
                in_use: AtomicBool::new(false),
            },
            CommandBufferUsage::SimultaneousUse => SubmitState::Concurrent,
            CommandBufferUsage::OneTimeSubmit => SubmitState::OneTime {
                already_submitted: AtomicBool::new(false),
            },
        };

        Ok(PrimaryAutoCommandBuffer {
            inner: self.inner.build()?,
            pool_alloc: self.pool_builder_alloc.into_alloc(),
            submit_state,
        })
    }
}

impl<P> AutoCommandBufferBuilder<SecondaryAutoCommandBuffer<P::Alloc>, P>
where
    P: CommandPoolBuilderAlloc,
{
    /// Builds the command buffer.
    #[inline]
    pub fn build(self) -> Result<SecondaryAutoCommandBuffer<P::Alloc>, BuildError> {
        if !self.query_state.is_empty() {
            return Err(AutoCommandBufferBuilderContextError::QueryIsActive.into());
        }

        let submit_state = match self.usage {
            CommandBufferUsage::MultipleSubmit => SubmitState::ExclusiveUse {
                in_use: AtomicBool::new(false),
            },
            CommandBufferUsage::SimultaneousUse => SubmitState::Concurrent,
            CommandBufferUsage::OneTimeSubmit => SubmitState::OneTime {
                already_submitted: AtomicBool::new(false),
            },
        };

        Ok(SecondaryAutoCommandBuffer {
            inner: self.inner.build()?,
            pool_alloc: self.pool_builder_alloc.into_alloc(),
            inheritance_info: self.inheritance_info.unwrap(),
            submit_state,
        })
    }
}

impl<L, P> AutoCommandBufferBuilder<L, P> {
    #[inline]
    pub(super) fn ensure_outside_render_pass(
        &self,
    ) -> Result<(), AutoCommandBufferBuilderContextError> {
        if self.render_pass_state.is_some() {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass);
        }

        Ok(())
    }

    #[inline]
    pub(super) fn ensure_inside_render_pass_inline(
        &self,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), AutoCommandBufferBuilderContextError> {
        let render_pass_state = self
            .render_pass_state
            .as_ref()
            .ok_or(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass)?;

        // Subpass must be for inline commands
        if render_pass_state.contents != SubpassContents::Inline {
            return Err(AutoCommandBufferBuilderContextError::WrongSubpassType);
        }

        match &render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(state) => {
                let pipeline_subpass = match pipeline.render_pass() {
                    PipelineRenderPassType::BeginRenderPass(subpass) => subpass,
                    PipelineRenderPassType::BeginRendering(_) => todo!(),
                };

                // Subpasses must be the same.
                if pipeline_subpass.index() != state.subpass.index() {
                    return Err(AutoCommandBufferBuilderContextError::WrongSubpassIndex);
                }

                // Render passes must be compatible.
                if !pipeline_subpass
                    .render_pass()
                    .is_compatible_with(&state.subpass.render_pass())
                {
                    return Err(AutoCommandBufferBuilderContextError::IncompatibleRenderPass);
                }
            }
            RenderPassStateType::BeginRendering(state) => {
                let pipeline_rendering_info = match pipeline.render_pass() {
                    PipelineRenderPassType::BeginRenderPass(_) => todo!(),
                    PipelineRenderPassType::BeginRendering(rendering_info) => rendering_info,
                };

                // TODO: checks
            }
            RenderPassStateType::Inherited => {
                match self
                    .inheritance_info
                    .as_ref()
                    .unwrap()
                    .render_pass
                    .as_ref()
                    .unwrap()
                {
                    CommandBufferInheritanceRenderPassType::BeginRenderPass(info) => {
                        let pipeline_subpass = match pipeline.render_pass() {
                            PipelineRenderPassType::BeginRenderPass(subpass) => subpass,
                            PipelineRenderPassType::BeginRendering(_) => todo!(),
                        };

                        // Subpasses must be the same.
                        if pipeline_subpass.index() != info.subpass.index() {
                            return Err(AutoCommandBufferBuilderContextError::WrongSubpassIndex);
                        }

                        // Render passes must be compatible.
                        if !pipeline_subpass
                            .render_pass()
                            .is_compatible_with(&info.subpass.render_pass())
                        {
                            return Err(
                                AutoCommandBufferBuilderContextError::IncompatibleRenderPass,
                            );
                        }
                    }
                    CommandBufferInheritanceRenderPassType::BeginRendering(_) => {
                        let pipeline_rendering_info = match pipeline.render_pass() {
                            PipelineRenderPassType::BeginRenderPass(_) => todo!(),
                            PipelineRenderPassType::BeginRendering(rendering_info) => {
                                rendering_info
                            }
                        };

                        // TODO: checks
                    }
                }
            }
        }

        Ok(())
    }

    #[inline]
    pub(super) fn queue_family(&self) -> QueueFamily {
        self.device()
            .physical_device()
            .queue_family_by_id(self.queue_family_id)
            .unwrap()
    }

    /// Returns the binding/setting state.
    #[inline]
    pub fn state(&self) -> CommandBufferState {
        self.inner.state()
    }
}

unsafe impl<L, P> DeviceOwned for AutoCommandBufferBuilder<L, P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

pub struct PrimaryAutoCommandBuffer<P = StandardCommandPoolAlloc> {
    inner: SyncCommandBuffer,
    pool_alloc: P, // Safety: must be dropped after `inner`

    // Tracks usage of the command buffer on the GPU.
    submit_state: SubmitState,
}

unsafe impl<P> DeviceOwned for PrimaryAutoCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<P> PrimaryCommandBuffer for PrimaryAutoCommandBuffer<P>
where
    P: CommandPoolAlloc,
{
    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer {
        self.inner.as_ref()
    }

    #[inline]
    fn lock_submit(
        &self,
        future: &dyn GpuFuture,
        queue: &Queue,
    ) -> Result<(), CommandBufferExecError> {
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                let was_already_submitted = already_submitted.swap(true, Ordering::SeqCst);
                if was_already_submitted {
                    return Err(CommandBufferExecError::OneTimeSubmitAlreadySubmitted);
                }
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let already_in_use = in_use.swap(true, Ordering::SeqCst);
                if already_in_use {
                    return Err(CommandBufferExecError::ExclusiveAlreadyInUse);
                }
            }
            SubmitState::Concurrent => (),
        };

        let err = match self.inner.lock_submit(future, queue) {
            Ok(()) => return Ok(()),
            Err(err) => err,
        };

        // If `self.inner.lock_submit()` failed, we revert action.
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                already_submitted.store(false, Ordering::SeqCst);
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                in_use.store(false, Ordering::SeqCst);
            }
            SubmitState::Concurrent => (),
        };

        Err(err)
    }

    #[inline]
    unsafe fn unlock(&self) {
        // Because of panic safety, we unlock the inner command buffer first.
        self.inner.unlock();

        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                debug_assert!(already_submitted.load(Ordering::SeqCst));
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let old_val = in_use.swap(false, Ordering::SeqCst);
                debug_assert!(old_val);
            }
            SubmitState::Concurrent => (),
        };
    }

    #[inline]
    fn check_buffer_access(
        &self,
        buffer: &UnsafeBuffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        self.inner
            .check_buffer_access(buffer, range, exclusive, queue)
    }

    #[inline]
    fn check_image_access(
        &self,
        image: &UnsafeImage,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        self.inner
            .check_image_access(image, range, exclusive, expected_layout, queue)
    }
}

pub struct SecondaryAutoCommandBuffer<P = StandardCommandPoolAlloc> {
    inner: SyncCommandBuffer,
    pool_alloc: P, // Safety: must be dropped after `inner`
    inheritance_info: CommandBufferInheritanceInfo,

    // Tracks usage of the command buffer on the GPU.
    submit_state: SubmitState,
}

unsafe impl<P> DeviceOwned for SecondaryAutoCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<P> SecondaryCommandBuffer for SecondaryAutoCommandBuffer<P>
where
    P: CommandPoolAlloc,
{
    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer {
        self.inner.as_ref()
    }

    #[inline]
    fn lock_record(&self) -> Result<(), CommandBufferExecError> {
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                let was_already_submitted = already_submitted.swap(true, Ordering::SeqCst);
                if was_already_submitted {
                    return Err(CommandBufferExecError::OneTimeSubmitAlreadySubmitted);
                }
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let already_in_use = in_use.swap(true, Ordering::SeqCst);
                if already_in_use {
                    return Err(CommandBufferExecError::ExclusiveAlreadyInUse);
                }
            }
            SubmitState::Concurrent => (),
        };

        Ok(())
    }

    #[inline]
    unsafe fn unlock(&self) {
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                debug_assert!(already_submitted.load(Ordering::SeqCst));
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let old_val = in_use.swap(false, Ordering::SeqCst);
                debug_assert!(old_val);
            }
            SubmitState::Concurrent => (),
        };
    }

    #[inline]
    fn inheritance_info(&self) -> &CommandBufferInheritanceInfo {
        &self.inheritance_info
    }

    #[inline]
    fn num_buffers(&self) -> usize {
        self.inner.num_buffers()
    }

    #[inline]
    fn buffer(
        &self,
        index: usize,
    ) -> Option<(
        &Arc<dyn BufferAccess>,
        Range<DeviceSize>,
        PipelineMemoryAccess,
    )> {
        self.inner.buffer(index)
    }

    #[inline]
    fn num_images(&self) -> usize {
        self.inner.num_images()
    }

    #[inline]
    fn image(
        &self,
        index: usize,
    ) -> Option<(
        &Arc<dyn ImageAccess>,
        &ImageSubresourceRange,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
    )> {
        self.inner.image(index)
    }
}

// Whether the command buffer can be submitted.
#[derive(Debug)]
enum SubmitState {
    // The command buffer was created with the "SimultaneousUse" flag. Can always be submitted at
    // any time.
    Concurrent,

    // The command buffer can only be submitted once simultaneously.
    ExclusiveUse {
        // True if the command buffer is current in use by the GPU.
        in_use: AtomicBool,
    },

    // The command buffer can only ever be submitted once.
    OneTime {
        // True if the command buffer has already been submitted once and can be no longer be
        // submitted.
        already_submitted: AtomicBool,
    },
}

macro_rules! err_gen {
    ($name:ident { $($err:ident,)+ }) => (
        #[derive(Debug, Clone)]
        pub enum $name {
            $(
                $err($err),
            )+
        }

        impl error::Error for $name {
            #[inline]
            fn source(&self) -> Option<&(dyn error::Error + 'static)> {
                match *self {
                    $(
                        $name::$err(ref err) => Some(err),
                    )+
                }
            }
        }

        impl fmt::Display for $name {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
                write!(fmt, "{}", match *self {
                    $(
                        $name::$err(_) => {
                            concat!("a ", stringify!($err))
                        }
                    )+
                })
            }
        }

        $(
            impl From<$err> for $name {
                #[inline]
                fn from(err: $err) -> $name {
                    $name::$err(err)
                }
            }
        )+
    );
}

err_gen!(BuildError {
    AutoCommandBufferBuilderContextError,
    OomError,
});

err_gen!(CopyQueryPoolResultsError {
    AutoCommandBufferBuilderContextError,
    CheckCopyQueryPoolResultsError,
    SyncCommandBufferBuilderError,
});

err_gen!(DispatchError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckDispatchError,
    SyncCommandBufferBuilderError,
});

err_gen!(DispatchIndirectError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckIndirectBufferError,
    CheckDispatchError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawIndexedError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    CheckIndexBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawIndirectError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    CheckIndirectBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawIndexedIndirectError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    CheckIndexBufferError,
    CheckIndirectBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(BeginQueryError {
    AutoCommandBufferBuilderContextError,
    CheckBeginQueryError,
});

err_gen!(EndQueryError {
    AutoCommandBufferBuilderContextError,
    CheckEndQueryError,
});

err_gen!(WriteTimestampError {
    AutoCommandBufferBuilderContextError,
    CheckWriteTimestampError,
});

err_gen!(ResetQueryPoolError {
    AutoCommandBufferBuilderContextError,
    CheckResetQueryPoolError,
});

#[derive(Debug, Copy, Clone)]
pub enum AutoCommandBufferBuilderContextError {
    /// Operation forbidden inside of a render pass.
    ForbiddenInsideRenderPass,
    /// Operation forbidden outside of a render pass.
    ForbiddenOutsideRenderPass,
    /// Tried to use a secondary command buffer with a specified framebuffer that is
    /// incompatible with the current framebuffer.
    IncompatibleFramebuffer,
    /// Tried to use a graphics pipeline or secondary command buffer whose render pass
    /// is incompatible with the current render pass.
    IncompatibleRenderPass,
    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,
    /// Tried to end a render pass with subpasses remaining, or tried to go to next subpass with no
    /// subpass remaining.
    NumSubpassesMismatch {
        /// Actual number of subpasses in the current render pass.
        actual: u32,
        /// Current subpass index before the failing command.
        current: u32,
    },
    /// A query is active that conflicts with the current operation.
    QueryIsActive,
    /// This query was not active.
    QueryNotActive,
    /// A query is active that is not included in the `inheritance` of the secondary command buffer.
    QueryNotInherited,
    /// Tried to use a graphics pipeline or secondary command buffer whose subpass index
    /// didn't match the current subpass index.
    WrongSubpassIndex,
    /// Tried to execute a secondary command buffer inside a subpass that only allows inline
    /// commands, or a draw command in a subpass that only allows secondary command buffers.
    WrongSubpassType,
}

impl error::Error for AutoCommandBufferBuilderContextError {}

impl fmt::Display for AutoCommandBufferBuilderContextError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass => {
                    "operation forbidden inside of a render pass"
                }
                AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass => {
                    "operation forbidden outside of a render pass"
                }
                AutoCommandBufferBuilderContextError::IncompatibleFramebuffer => {
                    "tried to use a secondary command buffer with a specified framebuffer that is \
                 incompatible with the current framebuffer"
                }
                AutoCommandBufferBuilderContextError::IncompatibleRenderPass => {
                    "tried to use a graphics pipeline or secondary command buffer whose render pass \
                  is incompatible with the current render pass"
                }
                AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily => {
                    "the queue family doesn't allow this operation"
                }
                AutoCommandBufferBuilderContextError::NumSubpassesMismatch { .. } => {
                    "tried to end a render pass with subpasses remaining, or tried to go to next \
                 subpass with no subpass remaining"
                }
                AutoCommandBufferBuilderContextError::QueryIsActive => {
                    "a query is active that conflicts with the current operation"
                }
                AutoCommandBufferBuilderContextError::QueryNotActive => {
                    "this query was not active"
                }
                AutoCommandBufferBuilderContextError::QueryNotInherited => {
                    "a query is active that is not included in the inheritance of the secondary command buffer"
                }
                AutoCommandBufferBuilderContextError::WrongSubpassIndex => {
                    "tried to use a graphics pipeline whose subpass index didn't match the current \
                 subpass index"
                }
                AutoCommandBufferBuilderContextError::WrongSubpassType => {
                    "tried to execute a secondary command buffer inside a subpass that only allows \
                 inline commands, or a draw command in a subpass that only allows secondary \
                 command buffers"
                }
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        buffer::{BufferUsage, CpuAccessibleBuffer},
        command_buffer::{BufferCopy, CopyBufferInfoTyped, CopyError, ExecuteCommandsError},
        device::{physical::PhysicalDevice, DeviceCreateInfo, QueueCreateInfo},
    };

    #[test]
    fn copy_buffer_dimensions() {
        let instance = instance!();

        let phys = match PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return,
        };

        let queue_family = match phys.queue_families().next() {
            Some(q) => q,
            None => return,
        };

        let (device, mut queues) = Device::new(
            phys,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let source = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            true,
            [1_u32, 2].iter().copied(),
        )
        .unwrap();

        let destination = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            true,
            [0_u32, 10, 20, 3, 4].iter().copied(),
        )
        .unwrap();

        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cbb.copy_buffer(CopyBufferInfoTyped {
            regions: [BufferCopy {
                src_offset: 0,
                dst_offset: 1,
                size: 2,
                ..Default::default()
            }]
            .into(),
            ..CopyBufferInfoTyped::buffers(source.clone(), destination.clone())
        })
        .unwrap();

        let cb = cbb.build().unwrap();

        let future = cb
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let result = destination.read().unwrap();

        assert_eq!(*result, [0_u32, 1, 2, 3, 4]);
    }

    #[test]
    fn secondary_nonconcurrent_conflict() {
        let (device, queue) = gfx_dev_and_queue!();

        // Make a secondary CB that doesn't support simultaneous use.
        let builder = AutoCommandBufferBuilder::secondary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
            Default::default(),
        )
        .unwrap();
        let secondary = Arc::new(builder.build().unwrap());

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();

            // Add the secondary a first time
            builder.execute_commands(secondary.clone()).unwrap();

            // Recording the same non-concurrent secondary command buffer twice into the same
            // primary is an error.
            assert!(matches!(
                builder.execute_commands(secondary.clone()),
                Err(ExecuteCommandsError::SyncCommandBufferBuilderError(
                    SyncCommandBufferBuilderError::ExecError(
                        CommandBufferExecError::ExclusiveAlreadyInUse
                    )
                ))
            ));
        }

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();
            builder.execute_commands(secondary.clone()).unwrap();
            let cb1 = builder.build().unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();

            // Recording the same non-concurrent secondary command buffer into multiple
            // primaries is an error.
            assert!(matches!(
                builder.execute_commands(secondary.clone()),
                Err(ExecuteCommandsError::SyncCommandBufferBuilderError(
                    SyncCommandBufferBuilderError::ExecError(
                        CommandBufferExecError::ExclusiveAlreadyInUse
                    )
                ))
            ));

            std::mem::drop(cb1);

            // Now that the first cb is dropped, we should be able to record.
            builder.execute_commands(secondary.clone()).unwrap();
        }
    }

    #[test]
    fn buffer_self_copy_overlapping() {
        let (device, queue) = gfx_dev_and_queue!();

        let source = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            true,
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfoTyped {
                regions: [BufferCopy {
                    src_offset: 0,
                    dst_offset: 2,
                    size: 2,
                    ..Default::default()
                }]
                .into(),
                ..CopyBufferInfoTyped::buffers(source.clone(), source.clone())
            })
            .unwrap();

        let cb = builder.build().unwrap();

        let future = cb
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let result = source.read().unwrap();

        assert_eq!(*result, [0_u32, 1, 0, 1]);
    }

    #[test]
    fn buffer_self_copy_not_overlapping() {
        let (device, queue) = gfx_dev_and_queue!();

        let source = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            true,
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        assert!(matches!(
            builder.copy_buffer(CopyBufferInfoTyped {
                regions: [BufferCopy {
                    src_offset: 0,
                    dst_offset: 1,
                    size: 2,
                    ..Default::default()
                }]
                .into(),
                ..CopyBufferInfoTyped::buffers(source.clone(), source.clone())
            }),
            Err(CopyError::OverlappingRegions {
                src_region_index: 0,
                dst_region_index: 0,
            })
        ));
    }
}
