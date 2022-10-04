// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    allocator::{
        standard::StandardCommandBufferAlloc, CommandBufferAlloc, CommandBufferAllocator,
        CommandBufferBuilderAlloc, StandardCommandBufferAllocator,
    },
    synced::{CommandBufferState, SyncCommandBuffer, SyncCommandBufferBuilder},
    sys::{CommandBufferBeginInfo, UnsafeCommandBuffer},
    CommandBufferExecError, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderPassInfo,
    CommandBufferInheritanceRenderPassType, CommandBufferLevel, CommandBufferUsage,
    PrimaryCommandBuffer, RenderingAttachmentInfo, SecondaryCommandBuffer, SubpassContents,
};
use crate::{
    buffer::{sys::UnsafeBuffer, BufferAccess},
    command_buffer::CommandBufferInheritanceRenderingInfo,
    device::{Device, DeviceOwned, Queue, QueueFamilyProperties},
    format::Format,
    image::{sys::UnsafeImage, ImageAccess, ImageLayout, ImageSubresourceRange},
    query::{QueryControlFlags, QueryType},
    render_pass::{Framebuffer, Subpass},
    sync::{AccessCheckError, AccessFlags, GpuFuture, PipelineMemoryAccess, PipelineStages},
    DeviceSize, OomError, RequirementNotMet, RequiresOneOf,
};
use std::{
    collections::HashMap,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    marker::PhantomData,
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// Note that command buffers allocated from `Arc<StandardCommandBufferAllocator>` don't implement
/// the `Send` and `Sync` traits. If you use this allocator, then the `AutoCommandBufferBuilder`
/// will not implement `Send` and `Sync` either. Once a command buffer is built, however, it *does*
/// implement `Send` and `Sync`.
pub struct AutoCommandBufferBuilder<L, A = Arc<StandardCommandBufferAllocator>>
where
    A: CommandBufferAllocator,
{
    pub(super) inner: SyncCommandBufferBuilder,
    builder_alloc: A::Builder, // Safety: must be dropped after `inner`

    // The index of the queue family that this command buffer is being created for.
    queue_family_index: u32,

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

// The state of the current render pass.
pub(super) struct RenderPassState {
    pub(super) contents: SubpassContents,
    pub(super) render_area_offset: [u32; 2],
    pub(super) render_area_extent: [u32; 2],
    pub(super) render_pass: RenderPassStateType,
    pub(super) view_mask: u32,
}

pub(super) enum RenderPassStateType {
    BeginRenderPass(BeginRenderPassState),
    BeginRendering(BeginRenderingState),
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
    pub(super) framebuffer: Option<Arc<Framebuffer>>,
}

pub(super) struct BeginRenderingState {
    pub(super) attachments: Option<BeginRenderingAttachments>,
    pub(super) color_attachment_formats: Vec<Option<Format>>,
    pub(super) depth_attachment_format: Option<Format>,
    pub(super) stencil_attachment_format: Option<Format>,
    pub(super) pipeline_used: bool,
}

pub(super) struct BeginRenderingAttachments {
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

impl<A> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, A>
where
    A: CommandBufferAllocator,
{
    /// Starts recording a primary command buffer.
    #[inline]
    pub fn primary(
        allocator: &A,
        queue_family_index: u32,
        usage: CommandBufferUsage,
    ) -> Result<
        AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<A::Alloc>, A>,
        CommandBufferBeginError,
    > {
        unsafe {
            AutoCommandBufferBuilder::begin(
                allocator,
                queue_family_index,
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

impl<A> AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, A>
where
    A: CommandBufferAllocator,
{
    /// Starts recording a secondary command buffer.
    #[inline]
    pub fn secondary(
        allocator: &A,
        queue_family_index: u32,
        usage: CommandBufferUsage,
        inheritance_info: CommandBufferInheritanceInfo,
    ) -> Result<
        AutoCommandBufferBuilder<SecondaryAutoCommandBuffer<A::Alloc>, A>,
        CommandBufferBeginError,
    > {
        unsafe {
            AutoCommandBufferBuilder::begin(
                allocator,
                queue_family_index,
                CommandBufferLevel::Secondary,
                CommandBufferBeginInfo {
                    usage,
                    inheritance_info: Some(inheritance_info),
                    _ne: crate::NonExhaustive(()),
                },
            )
        }
    }
}

impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    // Actual constructor. Private.
    //
    // `begin_info.inheritance_info` must match `level`.
    unsafe fn begin(
        allocator: &A,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<AutoCommandBufferBuilder<L, A>, CommandBufferBeginError> {
        Self::validate_begin(allocator.device(), queue_family_index, level, &begin_info)?;

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
                occlusion_query: _,
                query_statistics_flags: _,
                _ne: _,
            } = inheritance_info;

            if let Some(render_pass) = render_pass {
                // In a secondary command buffer, we don't know the render area yet, so use a
                // dummy value.
                let render_area_offset = [0, 0];
                let mut render_area_extent = [u32::MAX, u32::MAX];

                match render_pass {
                    CommandBufferInheritanceRenderPassType::BeginRenderPass(info) => {
                        if let Some(framebuffer) = &info.framebuffer {
                            // Still not exact, but it's a better upper bound.
                            render_area_extent = framebuffer.extent();
                        }

                        render_pass_state = Some(RenderPassState {
                            contents: SubpassContents::Inline,
                            render_area_offset,
                            render_area_extent,
                            render_pass: BeginRenderPassState {
                                subpass: info.subpass.clone(),
                                framebuffer: info.framebuffer.clone(),
                            }
                            .into(),
                            view_mask: info.subpass.subpass_desc().view_mask,
                        });
                    }
                    CommandBufferInheritanceRenderPassType::BeginRendering(info) => {
                        render_pass_state = Some(RenderPassState {
                            contents: SubpassContents::Inline,
                            render_area_offset,
                            render_area_extent,
                            render_pass: BeginRenderingState {
                                attachments: None,
                                color_attachment_formats: info.color_attachment_formats.clone(),
                                depth_attachment_format: info.depth_attachment_format,
                                stencil_attachment_format: info.stencil_attachment_format,
                                pipeline_used: false,
                            }
                            .into(),
                            view_mask: info.view_mask,
                        });
                    }
                }
            }
        }

        let builder_alloc = allocator
            .allocate(level, 1)?
            .next()
            .expect("Requested one command buffer from the command pool, but got zero.");

        let inner = SyncCommandBufferBuilder::new(builder_alloc.inner(), begin_info)?;

        Ok(AutoCommandBufferBuilder {
            inner,
            builder_alloc,
            queue_family_index,
            render_pass_state,
            query_state: HashMap::default(),
            inheritance_info,
            usage,
            _data: PhantomData,
        })
    }

    fn validate_begin(
        device: &Device,
        _queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: &CommandBufferBeginInfo,
    ) -> Result<(), CommandBufferBeginError> {
        let physical_device = device.physical_device();
        let properties = physical_device.properties();

        let &CommandBufferBeginInfo {
            usage: _,
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
                            return Err(CommandBufferBeginError::RequirementNotMet {
                                required_for: "`inheritance_info.render_pass` is `CommandBufferInheritanceRenderPassType::BeginRendering`, where `view_mask` is not `0`",
                                requires_one_of: RequiresOneOf {
                                    features: &["multiview"],
                                    ..Default::default()
                                },
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

                            // VUID-VkCommandBufferInheritanceRenderingInfo-pColorAttachmentFormats-parameter
                            format.validate_device(device)?;

                            // VUID-VkCommandBufferInheritanceRenderingInfo-pColorAttachmentFormats-06006
                            // Use unchecked, because all validation has been done above.
                            if !unsafe { physical_device.format_properties_unchecked(format) }
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
                            // VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-parameter
                            format.validate_device(device)?;

                            // VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06540
                            if !format.aspects().depth {
                                return Err(
                                    CommandBufferBeginError::DepthAttachmentFormatUsageNotSupported,
                                );
                            }

                            // VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06007
                            // Use unchecked, because all validation has been done above.
                            if !unsafe { physical_device.format_properties_unchecked(format) }
                                .potential_format_features()
                                .depth_stencil_attachment
                            {
                                return Err(
                                    CommandBufferBeginError::DepthAttachmentFormatUsageNotSupported,
                                );
                            }
                        }

                        if let Some(format) = stencil_attachment_format {
                            // VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-parameter
                            format.validate_device(device)?;

                            // VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-06541
                            if !format.aspects().stencil {
                                return Err(
                                    CommandBufferBeginError::StencilAttachmentFormatUsageNotSupported,
                                );
                            }

                            // VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-06199
                            // Use unchecked, because all validation has been done above.
                            if !unsafe { physical_device.format_properties_unchecked(format) }
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

                        // VUID-VkCommandBufferInheritanceRenderingInfo-rasterizationSamples-parameter
                        rasterization_samples.validate_device(device)?;
                    }
                }
            }

            if let Some(control_flags) = occlusion_query {
                // VUID-VkCommandBufferInheritanceInfo-queryFlags-00057
                control_flags.validate_device(device)?;

                // VUID-VkCommandBufferInheritanceInfo-occlusionQueryEnable-00056
                // VUID-VkCommandBufferInheritanceInfo-queryFlags-02788
                if !device.enabled_features().inherited_queries {
                    return Err(CommandBufferBeginError::RequirementNotMet {
                        required_for: "`inheritance_info.occlusion_query` is `Some`",
                        requires_one_of: RequiresOneOf {
                            features: &["inherited_queries"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-vkBeginCommandBuffer-commandBuffer-00052
                if control_flags.precise && !device.enabled_features().occlusion_query_precise {
                    return Err(CommandBufferBeginError::RequirementNotMet {
                        required_for:
                            "`inheritance_info.occlusion_query` is `Some(control_flags)`, where `control_flags.precise` is set",
                        requires_one_of: RequiresOneOf {
                            features: &["occlusion_query_precise"],
                            ..Default::default()
                        },
                    });
                }
            }

            // VUID-VkCommandBufferInheritanceInfo-pipelineStatistics-02789
            query_statistics_flags.validate_device(device)?;

            // VUID-VkCommandBufferInheritanceInfo-pipelineStatistics-00058
            if query_statistics_flags.count() > 0
                && !device.enabled_features().pipeline_statistics_query
            {
                return Err(CommandBufferBeginError::RequirementNotMet {
                    required_for: "`inheritance_info.query_statistics_flags` is not empty",
                    requires_one_of: RequiresOneOf {
                        features: &["pipeline_statistics_query"],
                        ..Default::default()
                    },
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

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
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

impl Error for CommandBufferBeginError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl Display for CommandBufferBeginError {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),

            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),

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

impl From<RequirementNotMet> for CommandBufferBeginError {
    #[inline]
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

impl<A> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<A::Alloc>, A>
where
    A: CommandBufferAllocator,
{
    /// Builds the command buffer.
    #[inline]
    pub fn build(self) -> Result<PrimaryAutoCommandBuffer<A::Alloc>, BuildError> {
        if self.render_pass_state.is_some() {
            return Err(BuildError::RenderPassActive);
        }

        if !self.query_state.is_empty() {
            return Err(BuildError::QueryActive);
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
            _alloc: self.builder_alloc.into_alloc(),
            submit_state,
        })
    }
}

impl<A> AutoCommandBufferBuilder<SecondaryAutoCommandBuffer<A::Alloc>, A>
where
    A: CommandBufferAllocator,
{
    /// Builds the command buffer.
    #[inline]
    pub fn build(self) -> Result<SecondaryAutoCommandBuffer<A::Alloc>, BuildError> {
        if !self.query_state.is_empty() {
            return Err(BuildError::QueryActive);
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
            _alloc: self.builder_alloc.into_alloc(),
            inheritance_info: self.inheritance_info.unwrap(),
            submit_state,
        })
    }
}

/// Error that can happen when building a command buffer.
#[derive(Clone, Debug)]
pub enum BuildError {
    OomError(OomError),

    /// A render pass is still active on the command buffer.
    RenderPassActive,

    /// A query is still active on the command buffer.
    QueryActive,
}

impl Error for BuildError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for BuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "out of memory"),
            Self::RenderPassActive => {
                write!(f, "a render pass is still active on the command buffer")
            }
            Self::QueryActive => write!(f, "a query is still active on the command buffer"),
        }
    }
}

impl From<OomError> for BuildError {
    #[inline]
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    #[inline]
    pub(super) fn queue_family_properties(&self) -> &QueueFamilyProperties {
        &self.device().physical_device().queue_family_properties()[self.queue_family_index as usize]
    }

    /// Returns the binding/setting state.
    #[inline]
    pub fn state(&self) -> CommandBufferState<'_> {
        self.inner.state()
    }
}

unsafe impl<L, A> DeviceOwned for AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

pub struct PrimaryAutoCommandBuffer<A = StandardCommandBufferAlloc> {
    inner: SyncCommandBuffer,
    _alloc: A, // Safety: must be dropped after `inner`

    // Tracks usage of the command buffer on the GPU.
    submit_state: SubmitState,
}

unsafe impl<P> DeviceOwned for PrimaryAutoCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<A> PrimaryCommandBuffer for PrimaryAutoCommandBuffer<A>
where
    A: CommandBufferAlloc,
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

pub struct SecondaryAutoCommandBuffer<A = StandardCommandBufferAlloc> {
    inner: SyncCommandBuffer,
    _alloc: A, // Safety: must be dropped after `inner`
    inheritance_info: CommandBufferInheritanceInfo,

    // Tracks usage of the command buffer on the GPU.
    submit_state: SubmitState,
}

unsafe impl<A> DeviceOwned for SecondaryAutoCommandBuffer<A> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<A> SecondaryCommandBuffer for SecondaryAutoCommandBuffer<A>
where
    A: CommandBufferAlloc,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        buffer::{BufferUsage, CpuAccessibleBuffer},
        command_buffer::{
            synced::SyncCommandBufferBuilderError, BufferCopy, CopyBufferInfoTyped, CopyError,
            ExecuteCommandsError,
        },
        device::{DeviceCreateInfo, QueueCreateInfo},
    };

    #[test]
    fn copy_buffer_dimensions() {
        let instance = instance!();

        let physical_device = match instance.enumerate_physical_devices().unwrap().next() {
            Some(p) => p,
            None => return,
        };

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let source = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                transfer_src: true,
                ..BufferUsage::empty()
            },
            true,
            [1_u32, 2].iter().copied(),
        )
        .unwrap();

        let destination = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                transfer_dst: true,
                ..BufferUsage::empty()
            },
            true,
            [0_u32, 10, 20, 3, 4].iter().copied(),
        )
        .unwrap();

        let allocator =
            StandardCommandBufferAllocator::new(device, queue.queue_family_index()).unwrap();
        let mut cbb = AutoCommandBufferBuilder::primary(
            &allocator,
            queue.queue_family_index(),
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
            ..CopyBufferInfoTyped::buffers(source, destination.clone())
        })
        .unwrap();

        let cb = cbb.build().unwrap();

        let future = cb
            .execute(queue)
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

        let allocator =
            StandardCommandBufferAllocator::new(device, queue.queue_family_index()).unwrap();

        // Make a secondary CB that doesn't support simultaneous use.
        let builder = AutoCommandBufferBuilder::secondary(
            &allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            Default::default(),
        )
        .unwrap();
        let secondary = Arc::new(builder.build().unwrap());

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                &allocator,
                queue.queue_family_index(),
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
                &allocator,
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();
            builder.execute_commands(secondary.clone()).unwrap();
            let cb1 = builder.build().unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                &allocator,
                queue.queue_family_index(),
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
            builder.execute_commands(secondary).unwrap();
        }
    }

    #[test]
    fn buffer_self_copy_overlapping() {
        let (device, queue) = gfx_dev_and_queue!();

        let source = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                transfer_src: true,
                transfer_dst: true,
                ..BufferUsage::empty()
            },
            true,
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let allocator =
            StandardCommandBufferAllocator::new(device, queue.queue_family_index()).unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            &allocator,
            queue.queue_family_index(),
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
            .execute(queue)
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
            BufferUsage {
                transfer_src: true,
                transfer_dst: true,
                ..BufferUsage::empty()
            },
            true,
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let allocator =
            StandardCommandBufferAllocator::new(device, queue.queue_family_index()).unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            &allocator,
            queue.queue_family_index(),
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
                ..CopyBufferInfoTyped::buffers(source.clone(), source)
            }),
            Err(CopyError::OverlappingRegions {
                src_region_index: 0,
                dst_region_index: 0,
            })
        ));
    }
}
