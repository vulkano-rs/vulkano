// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    CommandInfo, PrimaryAutoCommandBuffer, RenderPassCommand, Resource, ResourceUseRef2,
    SubmitState,
};
use crate::{
    buffer::{Buffer, IndexBuffer, Subbuffer},
    command_buffer::{
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
        sys::{CommandBufferBeginInfo, UnsafeCommandBuffer, UnsafeCommandBufferBuilder},
        CommandBufferBufferRangeUsage, CommandBufferBufferUsage, CommandBufferImageRangeUsage,
        CommandBufferImageUsage, CommandBufferInheritanceInfo,
        CommandBufferInheritanceRenderPassInfo, CommandBufferInheritanceRenderPassType,
        CommandBufferInheritanceRenderingInfo, CommandBufferLevel, CommandBufferResourcesUsage,
        CommandBufferUsage, RenderingInfo, ResourceUseRef, SecondaryAutoCommandBuffer,
        SecondaryCommandBufferBufferUsage, SecondaryCommandBufferImageUsage,
        SecondaryCommandBufferResourcesUsage, SubpassContents,
    },
    descriptor_set::{DescriptorSetResources, DescriptorSetWithOffsets},
    device::{Device, DeviceOwned, QueueFamilyProperties},
    format::FormatFeatures,
    image::{
        sys::Image, ImageAccess, ImageAspects, ImageLayout, ImageSubresourceRange,
        ImageViewAbstract,
    },
    pipeline::{
        graphics::{
            color_blend::LogicOp,
            depth_stencil::{CompareOp, StencilOps},
            input_assembly::PrimitiveTopology,
            rasterization::{CullMode, DepthBias, FrontFace, LineStipple},
            subpass::PipelineRenderingCreateInfo,
            viewport::{Scissor, Viewport},
        },
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    },
    query::{QueryControlFlags, QueryPool},
    range_map::RangeMap,
    range_set::RangeSet,
    render_pass::{Framebuffer, Subpass},
    sync::{
        AccessFlags, BufferMemoryBarrier, DependencyInfo, ImageMemoryBarrier,
        PipelineStageAccessFlags, PipelineStages,
    },
    DeviceSize, OomError, RequirementNotMet, RequiresOneOf,
};
use ahash::HashMap;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    collections::hash_map::Entry,
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    marker::PhantomData,
    mem::take,
    ops::{Range, RangeInclusive},
    sync::{atomic::AtomicBool, Arc},
};

/// Note that command buffers allocated from `StandardCommandBufferAllocator` don't implement
/// the `Send` and `Sync` traits. If you use this allocator, then the `AutoCommandBufferBuilder`
/// will not implement `Send` and `Sync` either. Once a command buffer is built, however, it *does*
/// implement `Send` and `Sync`.
pub struct AutoCommandBufferBuilder<L, A = StandardCommandBufferAllocator>
where
    A: CommandBufferAllocator,
{
    pub(in crate::command_buffer) inner: UnsafeCommandBufferBuilder<A>,
    commands: Vec<(
        CommandInfo,
        Box<dyn Fn(&mut UnsafeCommandBufferBuilder<A>) + Send + Sync + 'static>,
    )>,
    pub(in crate::command_buffer) builder_state: CommandBufferBuilderState,
    _data: PhantomData<L>,
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
    ) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<A>, A>, CommandBufferBeginError>
    {
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

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn primary_unchecked(
        allocator: &A,
        queue_family_index: u32,
        usage: CommandBufferUsage,
    ) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<A>, A>, OomError> {
        AutoCommandBufferBuilder::begin_unchecked(
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
    ) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer<A>, A>, CommandBufferBeginError>
    {
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

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn secondary_unchecked(
        allocator: &A,
        queue_family_index: u32,
        usage: CommandBufferUsage,
        inheritance_info: CommandBufferInheritanceInfo,
    ) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer<A>, A>, OomError> {
        AutoCommandBufferBuilder::begin_unchecked(
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

impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Actual constructor. Private.
    ///
    /// # Safety
    ///
    /// `begin_info.inheritance_info` must match `level`.
    unsafe fn begin(
        allocator: &A,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<AutoCommandBufferBuilder<L, A>, CommandBufferBeginError> {
        Self::validate_begin(allocator.device(), queue_family_index, level, &begin_info)?;

        unsafe {
            Ok(Self::begin_unchecked(
                allocator,
                queue_family_index,
                level,
                begin_info,
            )?)
        }
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
                                required_for: "`inheritance_info.render_pass` is \
                                    `CommandBufferInheritanceRenderPassType::BeginRendering`, \
                                    where `view_mask` is not `0`",
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
                                .intersects(FormatFeatures::COLOR_ATTACHMENT)
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
                            if !format.aspects().intersects(ImageAspects::DEPTH) {
                                return Err(
                                    CommandBufferBeginError::DepthAttachmentFormatUsageNotSupported,
                                );
                            }

                            // VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06007
                            // Use unchecked, because all validation has been done above.
                            if !unsafe { physical_device.format_properties_unchecked(format) }
                                .potential_format_features()
                                .intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
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
                            if !format.aspects().intersects(ImageAspects::STENCIL) {
                                return Err(
                                    CommandBufferBeginError::StencilAttachmentFormatUsageNotSupported,
                                );
                            }

                            // VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-06199
                            // Use unchecked, because all validation has been done above.
                            if !unsafe { physical_device.format_properties_unchecked(format) }
                                .potential_format_features()
                                .intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
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
                if control_flags.intersects(QueryControlFlags::PRECISE)
                    && !device.enabled_features().occlusion_query_precise
                {
                    return Err(CommandBufferBeginError::RequirementNotMet {
                        required_for: "`inheritance_info.occlusion_query` is \
                            `Some(control_flags)`, where `control_flags` contains \
                            `QueryControlFlags::PRECISE`",
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

    #[inline]
    unsafe fn begin_unchecked(
        allocator: &A,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<Self, OomError> {
        let &CommandBufferBeginInfo {
            usage: _,
            ref inheritance_info,
            _ne: _,
        } = &begin_info;

        let mut builder_state: CommandBufferBuilderState = Default::default();

        if let Some(inheritance_info) = &inheritance_info {
            let &CommandBufferInheritanceInfo {
                ref render_pass,
                occlusion_query: _,
                query_statistics_flags: _,
                _ne: _,
            } = inheritance_info;

            if let Some(render_pass) = render_pass {
                builder_state.render_pass = Some(RenderPassState::from_inheritance(render_pass));
            }
        }

        let inner =
            UnsafeCommandBufferBuilder::new(allocator, queue_family_index, level, begin_info)?;

        Ok(AutoCommandBufferBuilder {
            inner,
            commands: Vec::new(),
            builder_state,
            _data: PhantomData,
        })
    }

    unsafe fn end_unchecked(
        mut self,
    ) -> Result<
        (
            UnsafeCommandBuffer<A>,
            Vec<Box<dyn Fn(&mut UnsafeCommandBufferBuilder<A>) + Send + Sync + 'static>>,
            CommandBufferResourcesUsage,
            SecondaryCommandBufferResourcesUsage,
        ),
        CommandBufferBuildError,
    > {
        let mut auto_sync_state = AutoSyncState::new(
            self.device().clone(),
            self.inner.level(),
            self.inner
                .inheritance_info()
                .as_ref()
                .map_or(false, |info| info.render_pass.is_some()),
        );

        // Add barriers between the commands.
        for (command_info, _) in self.commands.iter() {
            auto_sync_state.add_command(command_info)?;
        }

        let (mut barriers, resources_usage, secondary_resources_usage) = auto_sync_state.build();
        let final_barrier_index = self.commands.len();

        // Record all the commands and barriers to the inner command buffer.
        for (command_index, (_, record_func)) in self.commands.iter().enumerate() {
            if let Some(barriers) = barriers.remove(&command_index) {
                for dependency_info in barriers {
                    unsafe {
                        self.inner.pipeline_barrier(&dependency_info);
                    }
                }
            }

            record_func(&mut self.inner);
        }

        // Record final barriers
        if let Some(final_barriers) = barriers.remove(&final_barrier_index) {
            for dependency_info in final_barriers {
                unsafe {
                    self.inner.pipeline_barrier(&dependency_info);
                }
            }
        }

        debug_assert!(barriers.is_empty());

        Ok((
            self.inner.build()?,
            self.commands
                .into_iter()
                .map(|(_, record_func)| record_func)
                .collect(),
            resources_usage,
            secondary_resources_usage,
        ))
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
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for CommandBufferBeginError {
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
                write!(f, "the framebuffer is not compatible with the render pass")
            }
            Self::MaxMultiviewViewCountExceeded { .. } => {
                write!(f, "the `max_multiview_view_count` limit has been exceeded")
            }
            Self::StencilAttachmentFormatUsageNotSupported => write!(
                f,
                "the stencil attachment has a format that does not support that usage",
            ),
        }
    }
}

impl From<OomError> for CommandBufferBeginError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<RequirementNotMet> for CommandBufferBeginError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

impl<A> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<A>, A>
where
    A: CommandBufferAllocator,
{
    /// Builds the command buffer.
    pub fn build(self) -> Result<Arc<PrimaryAutoCommandBuffer<A>>, CommandBufferBuildError> {
        if self.builder_state.render_pass.is_some() {
            return Err(CommandBufferBuildError::RenderPassActive);
        }

        if !self.builder_state.queries.is_empty() {
            return Err(CommandBufferBuildError::QueryActive);
        }

        let (inner, keep_alive_objects, resources_usage, _) = unsafe { self.end_unchecked()? };

        Ok(Arc::new(PrimaryAutoCommandBuffer {
            inner,
            _keep_alive_objects: keep_alive_objects,
            resources_usage,
            state: Mutex::new(Default::default()),
        }))
    }
}

impl<A> AutoCommandBufferBuilder<SecondaryAutoCommandBuffer<A>, A>
where
    A: CommandBufferAllocator,
{
    /// Builds the command buffer.
    pub fn build(self) -> Result<Arc<SecondaryAutoCommandBuffer<A>>, CommandBufferBuildError> {
        if !self.builder_state.queries.is_empty() {
            return Err(CommandBufferBuildError::QueryActive);
        }

        let submit_state = match self.inner.usage() {
            CommandBufferUsage::MultipleSubmit => SubmitState::ExclusiveUse {
                in_use: AtomicBool::new(false),
            },
            CommandBufferUsage::SimultaneousUse => SubmitState::Concurrent,
            CommandBufferUsage::OneTimeSubmit => SubmitState::OneTime {
                already_submitted: AtomicBool::new(false),
            },
        };

        let (inner, keep_alive_objects, _, resources_usage) = unsafe { self.end_unchecked()? };

        Ok(Arc::new(SecondaryAutoCommandBuffer {
            inner,
            _keep_alive_objects: keep_alive_objects,
            resources_usage,
            submit_state,
        }))
    }
}

/// Error that can happen when building a command buffer.
#[derive(Clone, Debug)]
pub enum CommandBufferBuildError {
    OomError(OomError),

    /// A render pass is still active on the command buffer.
    RenderPassActive,

    /// A query is still active on the command buffer.
    QueryActive,

    /// A conflict exists between two resources that cannot be resolved by a pipeline barrier.
    UnsolvableResourceConflict {
        current_use_ref: ResourceUseRef,
        previous_use_ref: ResourceUseRef,
    },
}

impl Error for CommandBufferBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for CommandBufferBuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "out of memory"),
            Self::RenderPassActive => {
                write!(f, "a render pass is still active on the command buffer")
            }
            Self::QueryActive => write!(f, "a query is still active on the command buffer"),
            Self::UnsolvableResourceConflict { .. } => write!(
                f,
                "a conflict exists between two resources that cannot be resolved by a \
                pipeline barrier"
            ),
        }
    }
}

impl From<OomError> for CommandBufferBuildError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    pub(in crate::command_buffer) fn queue_family_properties(&self) -> &QueueFamilyProperties {
        &self.device().physical_device().queue_family_properties()
            [self.inner.queue_family_index() as usize]
    }

    pub(in crate::command_buffer) fn add_command(
        &mut self,
        name: &'static str,
        used_resources: Vec<(ResourceUseRef2, Resource)>,
        record_func: impl Fn(&mut UnsafeCommandBufferBuilder<A>) + Send + Sync + 'static,
    ) {
        self.commands.push((
            CommandInfo {
                name,
                used_resources,
                render_pass: RenderPassCommand::None,
            },
            Box::new(record_func),
        ));
    }

    pub(in crate::command_buffer) fn add_render_pass_begin(
        &mut self,
        name: &'static str,
        used_resources: Vec<(ResourceUseRef2, Resource)>,
        record_func: impl Fn(&mut UnsafeCommandBufferBuilder<A>) + Send + Sync + 'static,
    ) {
        self.commands.push((
            CommandInfo {
                name,
                used_resources,
                render_pass: RenderPassCommand::Begin,
            },
            Box::new(record_func),
        ));
    }

    pub(in crate::command_buffer) fn add_render_pass_end(
        &mut self,
        name: &'static str,
        used_resources: Vec<(ResourceUseRef2, Resource)>,
        record_func: impl Fn(&mut UnsafeCommandBufferBuilder<A>) + Send + Sync + 'static,
    ) {
        self.commands.push((
            CommandInfo {
                name,
                used_resources,
                render_pass: RenderPassCommand::End,
            },
            Box::new(record_func),
        ));
    }
}

unsafe impl<L, A> DeviceOwned for AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

struct AutoSyncState {
    device: Arc<Device>,
    level: CommandBufferLevel,

    command_index: usize,
    barriers: HashMap<usize, Vec<DependencyInfo>>,
    pending_barrier: DependencyInfo,
    first_unflushed: usize,
    latest_render_pass_enter: Option<usize>,
    buffers: HashMap<Arc<Buffer>, RangeMap<DeviceSize, BufferState>>,
    images: HashMap<Arc<Image>, RangeMap<DeviceSize, ImageState>>,
    secondary_resources_usage: SecondaryCommandBufferResourcesUsage,
}

impl AutoSyncState {
    fn new(
        device: Arc<Device>,
        level: CommandBufferLevel,
        has_inherited_render_pass: bool,
    ) -> Self {
        Self {
            device,
            level,

            command_index: 0,
            pending_barrier: DependencyInfo::default(),
            barriers: Default::default(),
            first_unflushed: 0,
            latest_render_pass_enter: has_inherited_render_pass.then_some(0),
            buffers: HashMap::default(),
            images: HashMap::default(),
            secondary_resources_usage: Default::default(),
        }
    }

    fn build(
        mut self,
    ) -> (
        HashMap<usize, Vec<DependencyInfo>>,
        CommandBufferResourcesUsage,
        SecondaryCommandBufferResourcesUsage,
    ) {
        debug_assert!(self.latest_render_pass_enter.is_none() || self.pending_barrier.is_empty());

        // Add any pending barrier that might remain.
        self.barriers
            .entry(self.first_unflushed)
            .or_default()
            .push(take(&mut self.pending_barrier));

        // Add one last barrier to transition images to their desired final layout.
        if self.level == CommandBufferLevel::Primary {
            let mut final_barrier = DependencyInfo::default();

            for (image, range_map) in self.images.iter_mut() {
                for (range, state) in range_map
                    .iter_mut()
                    .filter(|(_range, state)| state.final_layout != state.current_layout)
                {
                    final_barrier
                        .image_memory_barriers
                        .push(ImageMemoryBarrier {
                            src_stages: PipelineStages::from(state.memory_access)
                                .into_supported(&self.device),
                            src_access: AccessFlags::from(state.memory_access)
                                .into_supported(&self.device),
                            dst_stages: PipelineStages::TOP_OF_PIPE,
                            dst_access: AccessFlags::empty(),
                            old_layout: state.current_layout,
                            new_layout: state.final_layout,
                            subresource_range: image.range_to_subresources(range.clone()),
                            ..ImageMemoryBarrier::image(image.clone())
                        });

                    state.is_written = true;
                }
            }

            self.barriers
                .entry(self.command_index)
                .or_default()
                .push(final_barrier);
        }

        let mut resources_usage = CommandBufferResourcesUsage {
            buffers: self
                .buffers
                .into_iter()
                .map(|(buffer, ranges)| CommandBufferBufferUsage {
                    buffer,
                    ranges: ranges
                        .into_iter()
                        .filter(|(_range, state)| !state.resource_uses.is_empty())
                        .map(|(range, state)| {
                            let first_use = state.resource_uses.into_iter().next();
                            (
                                range,
                                CommandBufferBufferRangeUsage {
                                    first_use,
                                    mutable: state.is_written,
                                },
                            )
                        })
                        .collect(),
                })
                .collect(),
            images: self
                .images
                .into_iter()
                .map(|(image, ranges)| CommandBufferImageUsage {
                    image,
                    ranges: ranges
                        .into_iter()
                        .filter(|(_range, state)| {
                            !state.resource_uses.is_empty()
                                || (self.level == CommandBufferLevel::Primary
                                    && state.current_layout != state.final_layout)
                        })
                        .map(|(range, mut state)| {
                            if self.level == CommandBufferLevel::Primary {
                                state.current_layout = state.final_layout;
                            }

                            let first_use = state.resource_uses.into_iter().next();
                            (
                                range,
                                CommandBufferImageRangeUsage {
                                    first_use,
                                    mutable: state.is_written,
                                    expected_layout: state.initial_layout,
                                    final_layout: state.current_layout,
                                },
                            )
                        })
                        .collect(),
                })
                .collect(),
            buffer_indices: Default::default(),
            image_indices: Default::default(),
        };

        resources_usage.buffer_indices = resources_usage
            .buffers
            .iter()
            .enumerate()
            .map(|(index, usage)| (usage.buffer.clone(), index))
            .collect();
        resources_usage.image_indices = resources_usage
            .images
            .iter()
            .enumerate()
            .map(|(index, usage)| (usage.image.clone(), index))
            .collect();

        (
            self.barriers,
            resources_usage,
            self.secondary_resources_usage,
        )
    }

    fn add_command(
        &mut self,
        command_info: &CommandInfo,
    ) -> Result<(), UnsolvableResourceConflict> {
        self.check_resource_conflicts(command_info)?;
        self.add_resources(command_info);

        match command_info.render_pass {
            RenderPassCommand::None => (),
            RenderPassCommand::Begin => {
                debug_assert!(self.latest_render_pass_enter.is_none());
                self.latest_render_pass_enter = Some(self.command_index);
            }
            RenderPassCommand::End => {
                debug_assert!(self.latest_render_pass_enter.is_some());
                self.latest_render_pass_enter = None;
            }
        }

        self.command_index += 1;

        Ok(())
    }

    fn check_resource_conflicts(
        &self,
        command_info: &CommandInfo,
    ) -> Result<(), UnsolvableResourceConflict> {
        let &CommandInfo {
            name: command_name,
            ref used_resources,
            ..
        } = command_info;

        for (use_ref, resource) in used_resources {
            match *resource {
                Resource::Buffer {
                    ref buffer,
                    ref range,
                    memory_access,
                } => {
                    if let Some(previous_use_ref) = self.find_buffer_conflict(
                        self.command_index,
                        buffer,
                        range.clone(),
                        memory_access,
                    ) {
                        return Err(UnsolvableResourceConflict {
                            current_use_ref: ResourceUseRef {
                                command_index: self.command_index,
                                command_name,
                                resource_in_command: use_ref.resource_in_command,
                                secondary_use_ref: use_ref.secondary_use_ref,
                            },
                            previous_use_ref,
                        });
                    }
                }
                Resource::Image {
                    ref image,
                    ref subresource_range,
                    memory_access,
                    start_layout,
                    end_layout,
                } => {
                    debug_assert!(memory_access.contains_write() || start_layout == end_layout);
                    debug_assert!(end_layout != ImageLayout::Undefined);
                    debug_assert!(end_layout != ImageLayout::Preinitialized);

                    if let Some(previous_use) = self.find_image_conflict(
                        image,
                        subresource_range.clone(),
                        memory_access,
                        start_layout,
                        end_layout,
                    ) {
                        return Err(UnsolvableResourceConflict {
                            current_use_ref: ResourceUseRef {
                                command_index: self.command_index,
                                command_name,
                                resource_in_command: use_ref.resource_in_command,
                                secondary_use_ref: use_ref.secondary_use_ref,
                            },
                            previous_use_ref: previous_use,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    fn find_buffer_conflict(
        &self,
        command_index: usize,
        buffer: &Subbuffer<[u8]>,
        mut range: Range<DeviceSize>,
        memory_access: PipelineStageAccessFlags,
    ) -> Option<ResourceUseRef> {
        // Barriers work differently in render passes, so if we're in one, we can only insert a
        // barrier before the start of the render pass.
        let last_allowed_barrier_index = self.latest_render_pass_enter.unwrap_or(command_index);

        range.start += buffer.offset();
        range.end += buffer.offset();

        let range_map = self.buffers.get(buffer.buffer())?;

        for (_range, state) in range_map
            .range(&range)
            .filter(|(_range, state)| !state.resource_uses.is_empty())
        {
            debug_assert!(state
                .resource_uses
                .iter()
                .all(|resource_use| resource_use.command_index <= command_index));

            if memory_access.contains_write() || state.memory_access.contains_write() {
                // If there is a resource use at a position beyond where we can insert a
                // barrier, then there is an unsolvable conflict.
                if let Some(&use_ref) = state
                    .resource_uses
                    .iter()
                    .find(|resource_use| resource_use.command_index >= last_allowed_barrier_index)
                {
                    return Some(use_ref);
                }
            }
        }

        None
    }

    fn find_image_conflict(
        &self,
        image: &dyn ImageAccess,
        subresource_range: ImageSubresourceRange,
        memory_access: PipelineStageAccessFlags,
        start_layout: ImageLayout,
        _end_layout: ImageLayout,
    ) -> Option<ResourceUseRef> {
        // Barriers work differently in render passes, so if we're in one, we can only insert a
        // barrier before the start of the render pass.
        let last_allowed_barrier_index =
            self.latest_render_pass_enter.unwrap_or(self.command_index);

        let inner = image.inner();
        let range_map = self.images.get(inner)?;

        for range in inner.iter_ranges(subresource_range) {
            for (_range, state) in range_map
                .range(&range)
                .filter(|(_range, state)| !state.resource_uses.is_empty())
            {
                debug_assert!(state
                    .resource_uses
                    .iter()
                    .all(|resource_use| resource_use.command_index <= self.command_index));

                // If the command expects the image to be undefined, then we can't
                // transition it, so use the current layout for both old and new layout.
                let start_layout = if start_layout == ImageLayout::Undefined {
                    state.current_layout
                } else {
                    start_layout
                };

                if memory_access.contains_write()
                    || state.memory_access.contains_write()
                    || state.current_layout != start_layout
                {
                    // If there is a resource use at a position beyond where we can insert a
                    // barrier, then there is an unsolvable conflict.
                    if let Some(&use_ref) = state.resource_uses.iter().find(|resource_use| {
                        resource_use.command_index >= last_allowed_barrier_index
                    }) {
                        return Some(use_ref);
                    }
                }
            }
        }

        None
    }

    /// Adds a command to be processed by the builder.
    ///
    /// The `resources` argument should contain each buffer or image used by the command.
    /// The function will take care of handling the pipeline barrier or flushing.
    ///
    /// - The index of the resource within the `resources` slice maps to the resource accessed
    ///   through `Command::buffer(..)` or `Command::image(..)`.
    /// - `PipelineMemoryAccess` must match the way the resource has been used.
    /// - `start_layout` and `end_layout` designate the image layout that the image is expected to
    ///   be in when the command starts, and the image layout that the image will be transitioned to
    ///   during the command. When it comes to buffers, you should pass `Undefined` for both.
    fn add_resources(&mut self, command_info: &CommandInfo) {
        let &CommandInfo {
            name: command_name,
            ref used_resources,
            ..
        } = command_info;

        for (use_ref, resource) in used_resources {
            match *resource {
                Resource::Buffer {
                    ref buffer,
                    ref range,
                    memory_access,
                } => {
                    self.add_buffer(
                        ResourceUseRef {
                            command_index: self.command_index,
                            command_name,
                            resource_in_command: use_ref.resource_in_command,
                            secondary_use_ref: use_ref.secondary_use_ref,
                        },
                        buffer.clone(),
                        range.clone(),
                        memory_access,
                    );
                }
                Resource::Image {
                    ref image,
                    ref subresource_range,
                    memory_access,
                    start_layout,
                    end_layout,
                } => {
                    self.add_image(
                        ResourceUseRef {
                            command_index: self.command_index,
                            command_name,
                            resource_in_command: use_ref.resource_in_command,
                            secondary_use_ref: use_ref.secondary_use_ref,
                        },
                        image.clone(),
                        subresource_range.clone(),
                        memory_access,
                        start_layout,
                        end_layout,
                    );
                }
            }
        }
    }

    fn add_buffer(
        &mut self,
        use_ref: ResourceUseRef,
        buffer: Subbuffer<[u8]>,
        mut range: Range<DeviceSize>,
        memory_access: PipelineStageAccessFlags,
    ) {
        self.secondary_resources_usage
            .buffers
            .push(SecondaryCommandBufferBufferUsage {
                use_ref,
                buffer: buffer.clone(),
                range: range.clone(),
                memory_access,
            });

        // Barriers work differently in render passes, so if we're in one, we can only insert a
        // barrier before the start of the render pass.
        let last_allowed_barrier_index =
            self.latest_render_pass_enter.unwrap_or(self.command_index);

        range.start += buffer.offset();
        range.end += buffer.offset();

        let range_map = self
            .buffers
            .entry(buffer.buffer().clone())
            .or_insert_with(|| {
                [(
                    0..buffer.buffer().size(),
                    BufferState {
                        resource_uses: Vec::new(),
                        memory_access: PipelineStageAccessFlags::empty(),
                        is_written: false,
                    },
                )]
                .into_iter()
                .collect()
            });
        range_map.split_at(&range.start);
        range_map.split_at(&range.end);

        for (range, state) in range_map.range_mut(&range) {
            if state.resource_uses.is_empty() {
                // This is the first time we use this resource range in this command buffer.
                state.resource_uses.push(use_ref);
                state.memory_access = memory_access;
                state.is_written = memory_access.contains_write();

                match self.level {
                    CommandBufferLevel::Primary => {
                        // To be safe, we insert a barrier for all stages and accesses before
                        // the first use, so that there are no hazards with any command buffer
                        // that was previously submitted to the same queue.
                        // This is rather overkill, but since command buffers don't know what
                        // will come before them, it's the only thing that works for now.
                        // TODO: come up with something better
                        let barrier = BufferMemoryBarrier {
                            src_stages: PipelineStages::ALL_COMMANDS,
                            src_access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                            dst_stages: PipelineStages::ALL_COMMANDS,
                            dst_access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                            range: range.clone(),
                            ..BufferMemoryBarrier::buffer(buffer.buffer().clone())
                        };

                        self.pending_barrier.buffer_memory_barriers.push(barrier);
                    }
                    CommandBufferLevel::Secondary => (),
                }
            } else {
                // This resource range was used before in this command buffer.

                // Find out if we have a collision with the pending commands.
                if memory_access.contains_write() || state.memory_access.contains_write() {
                    // Collision found between `latest_command_id` and `collision_cmd_id`.

                    // We now want to modify the current pipeline barrier in order to handle the
                    // collision. But since the pipeline barrier is going to be submitted before
                    // the flushed commands, it would be a mistake if `collision_cmd_id` hasn't
                    // been flushed yet.
                    if state
                        .resource_uses
                        .iter()
                        .any(|resource_use| resource_use.command_index >= self.first_unflushed)
                    {
                        // Add the pending barrier.
                        self.barriers
                            .entry(self.first_unflushed)
                            .or_default()
                            .push(take(&mut self.pending_barrier));
                        self.first_unflushed = last_allowed_barrier_index;
                    }

                    // Modify the pipeline barrier to handle the collision.
                    self.pending_barrier
                        .buffer_memory_barriers
                        .push(BufferMemoryBarrier {
                            src_stages: PipelineStages::from(state.memory_access)
                                .into_supported(&self.device),
                            src_access: AccessFlags::from(state.memory_access)
                                .into_supported(&self.device),
                            dst_stages: PipelineStages::from(memory_access)
                                .into_supported(&self.device),
                            dst_access: AccessFlags::from(memory_access)
                                .into_supported(&self.device),
                            range: range.clone(),
                            ..BufferMemoryBarrier::buffer(buffer.buffer().clone())
                        });

                    // Update state.
                    state.memory_access = memory_access;
                    state.is_written = true;
                } else {
                    // There is no collision. Simply merge the accesses.
                    state.memory_access |= memory_access;
                }

                state.resource_uses.push(use_ref);
            }
        }
    }

    fn add_image(
        &mut self,
        use_ref: ResourceUseRef,
        image: Arc<dyn ImageAccess>,
        mut subresource_range: ImageSubresourceRange,
        memory_access: PipelineStageAccessFlags,
        start_layout: ImageLayout,
        end_layout: ImageLayout,
    ) {
        self.secondary_resources_usage
            .images
            .push(SecondaryCommandBufferImageUsage {
                use_ref,
                image: image.clone(),
                subresource_range: subresource_range.clone(),
                memory_access,
                start_layout,
                end_layout,
            });

        // Barriers work differently in render passes, so if we're in one, we can only insert a
        // barrier before the start of the render pass.
        let last_allowed_barrier_index =
            self.latest_render_pass_enter.unwrap_or(self.command_index);

        let inner = image.inner();

        // VUID-VkImageMemoryBarrier2-image-03320
        if !self
            .device
            .enabled_features()
            .separate_depth_stencil_layouts
            && image
                .format()
                .aspects()
                .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
        {
            subresource_range.aspects = ImageAspects::DEPTH | ImageAspects::STENCIL;
        }

        let range_map = self.images.entry(inner.clone()).or_insert_with(|| {
            [(
                0..inner.range_size(),
                match self.level {
                    CommandBufferLevel::Primary => {
                        // In a primary command buffer, the initial layout is determined
                        // by the image.
                        let initial_layout = if !image.is_layout_initialized() {
                            unsafe {
                                image.layout_initialized();
                            }

                            image.initial_layout()
                        } else {
                            image.initial_layout_requirement()
                        };

                        ImageState {
                            resource_uses: Vec::new(),
                            memory_access: PipelineStageAccessFlags::empty(),
                            is_written: false,
                            initial_layout,
                            current_layout: initial_layout,
                            final_layout: image.final_layout_requirement(),
                        }
                    }
                    CommandBufferLevel::Secondary => {
                        // In a secondary command buffer, the initial layout is the layout
                        // of the first use.
                        ImageState {
                            resource_uses: Vec::new(),
                            memory_access: PipelineStageAccessFlags::empty(),
                            is_written: false,
                            initial_layout: ImageLayout::Undefined,
                            current_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::Undefined,
                        }
                    }
                },
            )]
            .into_iter()
            .collect()
        });

        for range in inner.iter_ranges(subresource_range) {
            range_map.split_at(&range.start);
            range_map.split_at(&range.end);

            for (range, state) in range_map.range_mut(&range) {
                if state.resource_uses.is_empty() {
                    // This is the first time we use this resource range in this command buffer.

                    debug_assert_eq!(state.initial_layout, state.current_layout);

                    state.resource_uses.push(use_ref);
                    state.memory_access = memory_access;
                    state.is_written = memory_access.contains_write();
                    state.current_layout = end_layout;

                    match self.level {
                        CommandBufferLevel::Primary => {
                            // To be safe, we insert a barrier for all stages and accesses before
                            // the first use, so that there are no hazards with any command buffer
                            // that was previously submitted to the same queue.
                            // This is rather overkill, but since command buffers don't know what
                            // will come before them, it's the only thing that works for now.
                            // TODO: come up with something better
                            let mut barrier = ImageMemoryBarrier {
                                src_stages: PipelineStages::ALL_COMMANDS,
                                src_access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                                dst_stages: PipelineStages::ALL_COMMANDS,
                                dst_access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                                old_layout: state.initial_layout,
                                new_layout: start_layout,
                                subresource_range: inner.range_to_subresources(range.clone()),
                                ..ImageMemoryBarrier::image(inner.clone())
                            };

                            // If the `new_layout` is Undefined or Preinitialized, this requires
                            // special handling. With the synchronization2 feature enabled, these
                            // values are permitted, as long as `new_layout` equals `old_layout`.
                            // Without synchronization2, these values are never permitted.
                            match barrier.new_layout {
                                ImageLayout::Undefined => {
                                    // If the command expects Undefined, that really means it
                                    // doesn't care about the layout at all and anything is valid.
                                    // We try to keep the old layout, or do a "dummy" transition to
                                    // the image's final layout if that isn't possible.
                                    barrier.new_layout = if !self
                                        .device
                                        .enabled_features()
                                        .synchronization2
                                        && matches!(
                                            barrier.old_layout,
                                            ImageLayout::Undefined | ImageLayout::Preinitialized
                                        ) {
                                        image.final_layout_requirement()
                                    } else {
                                        barrier.old_layout
                                    };
                                }
                                ImageLayout::Preinitialized => {
                                    // TODO: put this in find_image_conflict instead?

                                    // The image must be in the Preinitialized layout already, we
                                    // can't transition it.
                                    if state.initial_layout != ImageLayout::Preinitialized {
                                        panic!(
                                            "The command requires the `Preinitialized layout`, but
                                            the initial layout of the image is not `Preinitialized`"
                                        );
                                    }

                                    // The image is in the Preinitialized layout, but we can't keep
                                    // that layout because of the limitations of
                                    // pre-synchronization2 barriers. So an error is all we can do.
                                    if !self.device.enabled_features().synchronization2 {
                                        panic!(
                                            "The command requires the `Preinitialized` layout, \
                                            but this is not allowed in pipeline barriers without
                                            the `synchronization2` feature enabled"
                                        );
                                    }
                                }
                                _ => (),
                            }

                            // A layout transition is a write, so if we perform one, we
                            // need exclusive access.
                            if barrier.old_layout != barrier.new_layout {
                                state.is_written = true;
                            }

                            self.pending_barrier.image_memory_barriers.push(barrier);
                        }
                        CommandBufferLevel::Secondary => {
                            state.initial_layout = start_layout;
                        }
                    }
                } else {
                    // This resource range was used before in this command buffer.

                    // If the command expects the image to be undefined, then we can't
                    // transition it, so use the current layout for both old and new layout.
                    let start_layout = if start_layout == ImageLayout::Undefined {
                        state.current_layout
                    } else {
                        start_layout
                    };

                    // Find out if we have a collision with the pending commands.
                    if memory_access.contains_write()
                        || state.memory_access.contains_write()
                        || state.current_layout != start_layout
                    {
                        // Collision found between `latest_command_id` and `collision_cmd_id`.

                        // We now want to modify the current pipeline barrier in order to handle the
                        // collision. But since the pipeline barrier is going to be submitted before
                        // the flushed commands, it would be a mistake if `collision_cmd_id` hasn't
                        // been flushed yet.
                        if state
                            .resource_uses
                            .iter()
                            .any(|resource_use| resource_use.command_index >= self.first_unflushed)
                            || state.current_layout != start_layout
                        {
                            // Add the pending barrier.
                            self.barriers
                                .entry(self.first_unflushed)
                                .or_default()
                                .push(take(&mut self.pending_barrier));
                            self.first_unflushed = last_allowed_barrier_index;
                        }

                        // Modify the pipeline barrier to handle the collision.
                        self.pending_barrier
                            .image_memory_barriers
                            .push(ImageMemoryBarrier {
                                src_stages: PipelineStages::from(state.memory_access)
                                    .into_supported(&self.device),
                                src_access: AccessFlags::from(state.memory_access)
                                    .into_supported(&self.device),
                                dst_stages: PipelineStages::from(memory_access)
                                    .into_supported(&self.device),
                                dst_access: AccessFlags::from(memory_access)
                                    .into_supported(&self.device),
                                old_layout: state.current_layout,
                                new_layout: start_layout,
                                subresource_range: inner.range_to_subresources(range.clone()),
                                ..ImageMemoryBarrier::image(inner.clone())
                            });

                        // Update state.
                        state.memory_access = memory_access;
                        state.is_written = true;

                        if memory_access.contains_write() || end_layout != ImageLayout::Undefined {
                            state.current_layout = end_layout;
                        }
                    } else {
                        // There is no collision. Simply merge the accesses.
                        state.memory_access |= memory_access;
                    }

                    state.resource_uses.push(use_ref);
                }
            }
        }
    }
}

/// Error returned if the builder detects that there's an unsolvable conflict.
#[derive(Clone, Debug)]
struct UnsolvableResourceConflict {
    current_use_ref: ResourceUseRef,
    previous_use_ref: ResourceUseRef,
}

impl Error for UnsolvableResourceConflict {}

impl Display for UnsolvableResourceConflict {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(f, "unsolvable resource conflict")
    }
}

impl From<UnsolvableResourceConflict> for CommandBufferBuildError {
    fn from(value: UnsolvableResourceConflict) -> Self {
        let UnsolvableResourceConflict {
            current_use_ref,
            previous_use_ref,
        } = value;

        Self::UnsolvableResourceConflict {
            current_use_ref,
            previous_use_ref,
        }
    }
}

// State of a resource during the building of the command buffer.
#[derive(Clone, PartialEq, Eq)]
struct BufferState {
    // Lists every use of the resource.
    resource_uses: Vec<ResourceUseRef>,

    // Memory accesses performed since the last barrier.
    memory_access: PipelineStageAccessFlags,

    // True if the resource was written to at any point during the command buffer.
    // Also true if an image layout transition or queue transfer has been performed.
    is_written: bool,
}

// State of a resource during the building of the command buffer.
#[derive(Clone, PartialEq, Eq)]
struct ImageState {
    // Lists every use of the resource.
    resource_uses: Vec<ResourceUseRef>,

    // Memory accesses performed since the last barrier.
    memory_access: PipelineStageAccessFlags,

    // True if the resource was written to at any point during the command buffer.
    // Also true if an image layout transition or queue transfer has been performed.
    is_written: bool,

    // The layout that the image range must have when this command buffer is executed.
    // Can be `Undefined` if we don't care.
    initial_layout: ImageLayout,

    // Current layout at this stage of the building.
    current_layout: ImageLayout,

    // The layout that the image range will have at the end of the command buffer.
    // This is only used for primary command buffers.
    final_layout: ImageLayout,
}

/// Holds the current binding and setting state.
#[derive(Default)]
pub(in crate::command_buffer) struct CommandBufferBuilderState {
    // Render pass
    pub(in crate::command_buffer) render_pass: Option<RenderPassState>,

    // Bind/push
    pub(in crate::command_buffer) descriptor_sets: HashMap<PipelineBindPoint, DescriptorSetState>,
    pub(in crate::command_buffer) index_buffer: Option<IndexBuffer>,
    pub(in crate::command_buffer) pipeline_compute: Option<Arc<ComputePipeline>>,
    pub(in crate::command_buffer) pipeline_graphics: Option<Arc<GraphicsPipeline>>,
    pub(in crate::command_buffer) vertex_buffers: HashMap<u32, Subbuffer<[u8]>>,
    pub(in crate::command_buffer) push_constants: RangeSet<u32>,
    pub(in crate::command_buffer) push_constants_pipeline_layout: Option<Arc<PipelineLayout>>,

    // Dynamic state
    pub(in crate::command_buffer) blend_constants: Option<[f32; 4]>,
    pub(in crate::command_buffer) color_write_enable: Option<SmallVec<[bool; 4]>>,
    pub(in crate::command_buffer) cull_mode: Option<CullMode>,
    pub(in crate::command_buffer) depth_bias: Option<DepthBias>,
    pub(in crate::command_buffer) depth_bias_enable: Option<bool>,
    pub(in crate::command_buffer) depth_bounds: Option<RangeInclusive<f32>>,
    pub(in crate::command_buffer) depth_bounds_test_enable: Option<bool>,
    pub(in crate::command_buffer) depth_compare_op: Option<CompareOp>,
    pub(in crate::command_buffer) depth_test_enable: Option<bool>,
    pub(in crate::command_buffer) depth_write_enable: Option<bool>,
    pub(in crate::command_buffer) discard_rectangle: HashMap<u32, Scissor>,
    pub(in crate::command_buffer) front_face: Option<FrontFace>,
    pub(in crate::command_buffer) line_stipple: Option<LineStipple>,
    pub(in crate::command_buffer) line_width: Option<f32>,
    pub(in crate::command_buffer) logic_op: Option<LogicOp>,
    pub(in crate::command_buffer) patch_control_points: Option<u32>,
    pub(in crate::command_buffer) primitive_restart_enable: Option<bool>,
    pub(in crate::command_buffer) primitive_topology: Option<PrimitiveTopology>,
    pub(in crate::command_buffer) rasterizer_discard_enable: Option<bool>,
    pub(in crate::command_buffer) scissor: HashMap<u32, Scissor>,
    pub(in crate::command_buffer) scissor_with_count: Option<SmallVec<[Scissor; 2]>>,
    pub(in crate::command_buffer) stencil_compare_mask: StencilStateDynamic,
    pub(in crate::command_buffer) stencil_op: StencilOpStateDynamic,
    pub(in crate::command_buffer) stencil_reference: StencilStateDynamic,
    pub(in crate::command_buffer) stencil_test_enable: Option<bool>,
    pub(in crate::command_buffer) stencil_write_mask: StencilStateDynamic,
    pub(in crate::command_buffer) viewport: HashMap<u32, Viewport>,
    pub(in crate::command_buffer) viewport_with_count: Option<SmallVec<[Viewport; 2]>>,

    // Active queries
    pub(in crate::command_buffer) queries: HashMap<ash::vk::QueryType, QueryState>,
}

impl CommandBufferBuilderState {
    pub(in crate::command_buffer) fn reset_non_render_pass_states(&mut self) {
        *self = Self {
            render_pass: take(&mut self.render_pass),
            ..Default::default()
        }
    }

    pub(in crate::command_buffer) fn reset_dynamic_states(
        &mut self,
        states: impl IntoIterator<Item = DynamicState>,
    ) {
        for state in states {
            match state {
                DynamicState::BlendConstants => self.blend_constants = None,
                DynamicState::ColorWriteEnable => self.color_write_enable = None,
                DynamicState::CullMode => self.cull_mode = None,
                DynamicState::DepthBias => self.depth_bias = None,
                DynamicState::DepthBiasEnable => self.depth_bias_enable = None,
                DynamicState::DepthBounds => self.depth_bounds = None,
                DynamicState::DepthBoundsTestEnable => self.depth_bounds_test_enable = None,
                DynamicState::DepthCompareOp => self.depth_compare_op = None,
                DynamicState::DepthTestEnable => self.depth_test_enable = None,
                DynamicState::DepthWriteEnable => self.depth_write_enable = None,
                DynamicState::DiscardRectangle => self.discard_rectangle.clear(),
                DynamicState::ExclusiveScissor => (), // TODO;
                DynamicState::FragmentShadingRate => (), // TODO:
                DynamicState::FrontFace => self.front_face = None,
                DynamicState::LineStipple => self.line_stipple = None,
                DynamicState::LineWidth => self.line_width = None,
                DynamicState::LogicOp => self.logic_op = None,
                DynamicState::PatchControlPoints => self.patch_control_points = None,
                DynamicState::PrimitiveRestartEnable => self.primitive_restart_enable = None,
                DynamicState::PrimitiveTopology => self.primitive_topology = None,
                DynamicState::RasterizerDiscardEnable => self.rasterizer_discard_enable = None,
                DynamicState::RayTracingPipelineStackSize => (), // TODO:
                DynamicState::SampleLocations => (),             // TODO:
                DynamicState::Scissor => self.scissor.clear(),
                DynamicState::ScissorWithCount => self.scissor_with_count = None,
                DynamicState::StencilCompareMask => self.stencil_compare_mask = Default::default(),
                DynamicState::StencilOp => self.stencil_op = Default::default(),
                DynamicState::StencilReference => self.stencil_reference = Default::default(),
                DynamicState::StencilTestEnable => self.stencil_test_enable = None,
                DynamicState::StencilWriteMask => self.stencil_write_mask = Default::default(),
                DynamicState::VertexInput => (), // TODO:
                DynamicState::VertexInputBindingStride => (), // TODO:
                DynamicState::Viewport => self.viewport.clear(),
                DynamicState::ViewportCoarseSampleOrder => (), // TODO:
                DynamicState::ViewportShadingRatePalette => (), // TODO:
                DynamicState::ViewportWScaling => (),          // TODO:
                DynamicState::ViewportWithCount => self.viewport_with_count = None,
                DynamicState::TessellationDomainOrigin => (), // TODO:
                DynamicState::DepthClampEnable => (),         // TODO:
                DynamicState::PolygonMode => (),              // TODO:
                DynamicState::RasterizationSamples => (),     // TODO:
                DynamicState::SampleMask => (),               // TODO:
                DynamicState::AlphaToCoverageEnable => (),    // TODO:
                DynamicState::AlphaToOneEnable => (),         // TODO:
                DynamicState::LogicOpEnable => (),            // TODO:
                DynamicState::ColorBlendEnable => (),         // TODO:
                DynamicState::ColorBlendEquation => (),       // TODO:
                DynamicState::ColorWriteMask => (),           // TODO:
                DynamicState::RasterizationStream => (),      // TODO:
                DynamicState::ConservativeRasterizationMode => (), // TODO:
                DynamicState::ExtraPrimitiveOverestimationSize => (), // TODO:
                DynamicState::DepthClipEnable => (),          // TODO:
                DynamicState::SampleLocationsEnable => (),    // TODO:
                DynamicState::ColorBlendAdvanced => (),       // TODO:
                DynamicState::ProvokingVertexMode => (),      // TODO:
                DynamicState::LineRasterizationMode => (),    // TODO:
                DynamicState::LineStippleEnable => (),        // TODO:
                DynamicState::DepthClipNegativeOneToOne => (), // TODO:
                DynamicState::ViewportWScalingEnable => (),   // TODO:
                DynamicState::ViewportSwizzle => (),          // TODO:
                DynamicState::CoverageToColorEnable => (),    // TODO:
                DynamicState::CoverageToColorLocation => (),  // TODO:
                DynamicState::CoverageModulationMode => (),   // TODO:
                DynamicState::CoverageModulationTableEnable => (), // TODO:
                DynamicState::CoverageModulationTable => (),  // TODO:
                DynamicState::ShadingRateImageEnable => (),   // TODO:
                DynamicState::RepresentativeFragmentTestEnable => (), // TODO:
                DynamicState::CoverageReductionMode => (),    // TODO:
            }
        }
    }

    pub(in crate::command_buffer) fn invalidate_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        first_set: u32,
        num_descriptor_sets: u32,
    ) -> &mut DescriptorSetState {
        match self.descriptor_sets.entry(pipeline_bind_point) {
            Entry::Vacant(entry) => entry.insert(DescriptorSetState {
                descriptor_sets: Default::default(),
                pipeline_layout,
            }),
            Entry::Occupied(entry) => {
                let state = entry.into_mut();

                let invalidate_from = if state.pipeline_layout == pipeline_layout {
                    // If we're still using the exact same layout, then of course it's compatible.
                    None
                } else if state.pipeline_layout.push_constant_ranges()
                    != pipeline_layout.push_constant_ranges()
                {
                    // If the push constant ranges don't match,
                    // all bound descriptor sets are disturbed.
                    Some(0)
                } else {
                    // Find the first descriptor set layout in the current pipeline layout that
                    // isn't compatible with the corresponding set in the new pipeline layout.
                    // If an incompatible set was found, all bound sets from that slot onwards will
                    // be disturbed.
                    let current_layouts = state.pipeline_layout.set_layouts();
                    let new_layouts = pipeline_layout.set_layouts();
                    let max = (current_layouts.len() as u32).min(first_set + num_descriptor_sets);
                    (0..max).find(|&num| {
                        let num = num as usize;
                        !current_layouts[num].is_compatible_with(&new_layouts[num])
                    })
                };

                if let Some(invalidate_from) = invalidate_from {
                    // Remove disturbed sets and set new pipeline layout.
                    state
                        .descriptor_sets
                        .retain(|&num, _| num < invalidate_from);
                    state.pipeline_layout = pipeline_layout;
                } else if (first_set + num_descriptor_sets) as usize
                    >= state.pipeline_layout.set_layouts().len()
                {
                    // New layout is a superset of the old one.
                    state.pipeline_layout = pipeline_layout;
                }

                state
            }
        }
    }
}

pub(in crate::command_buffer) struct RenderPassState {
    pub(in crate::command_buffer) contents: SubpassContents,
    pub(in crate::command_buffer) render_area_offset: [u32; 2],
    pub(in crate::command_buffer) render_area_extent: [u32; 2],

    pub(in crate::command_buffer) rendering_info: PipelineRenderingCreateInfo,
    pub(in crate::command_buffer) attachments: Option<RenderPassStateAttachments>,

    pub(in crate::command_buffer) render_pass: RenderPassStateType,
}

impl RenderPassState {
    pub(in crate::command_buffer) fn from_inheritance(
        render_pass: &CommandBufferInheritanceRenderPassType,
    ) -> Self {
        match render_pass {
            CommandBufferInheritanceRenderPassType::BeginRenderPass(info) => {
                RenderPassState {
                    contents: SubpassContents::Inline,
                    render_area_offset: [0, 0],
                    render_area_extent: (info.framebuffer.as_ref())
                        // Still not exact, but it's a better upper bound.
                        .map_or([u32::MAX, u32::MAX], |framebuffer| framebuffer.extent()),

                    rendering_info: PipelineRenderingCreateInfo::from_subpass(&info.subpass),
                    attachments: info.framebuffer.as_ref().map(|framebuffer| {
                        RenderPassStateAttachments::from_subpass(&info.subpass, framebuffer)
                    }),

                    render_pass: BeginRenderPassState {
                        subpass: info.subpass.clone(),
                        framebuffer: info.framebuffer.clone(),
                    }
                    .into(),
                }
            }
            CommandBufferInheritanceRenderPassType::BeginRendering(info) => RenderPassState {
                contents: SubpassContents::Inline,
                render_area_offset: [0, 0],
                render_area_extent: [u32::MAX, u32::MAX],

                rendering_info: PipelineRenderingCreateInfo::from_inheritance_rendering_info(info),
                attachments: None,

                render_pass: BeginRenderingState {
                    pipeline_used: false,
                }
                .into(),
            },
        }
    }
}

pub(in crate::command_buffer) enum RenderPassStateType {
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

pub(in crate::command_buffer) struct BeginRenderPassState {
    pub(in crate::command_buffer) subpass: Subpass,
    pub(in crate::command_buffer) framebuffer: Option<Arc<Framebuffer>>,
}

pub(in crate::command_buffer) struct BeginRenderingState {
    pub(in crate::command_buffer) pipeline_used: bool,
}

pub(in crate::command_buffer) struct RenderPassStateAttachments {
    pub(in crate::command_buffer) color_attachments: Vec<Option<RenderPassStateAttachmentInfo>>,
    pub(in crate::command_buffer) depth_attachment: Option<RenderPassStateAttachmentInfo>,
    pub(in crate::command_buffer) stencil_attachment: Option<RenderPassStateAttachmentInfo>,
}

impl RenderPassStateAttachments {
    pub(in crate::command_buffer) fn from_subpass(
        subpass: &Subpass,
        framebuffer: &Framebuffer,
    ) -> Self {
        let subpass_desc = subpass.subpass_desc();
        let rp_attachments = subpass.render_pass().attachments();
        let fb_attachments = framebuffer.attachments();

        Self {
            color_attachments: (subpass_desc.color_attachments.iter())
                .zip(subpass_desc.color_resolve_attachments.iter())
                .map(|(color_attachment, color_resolve_attachment)| {
                    (color_attachment.as_ref()).map(|color_attachment| {
                        RenderPassStateAttachmentInfo {
                            image_view: fb_attachments[color_attachment.attachment as usize]
                                .clone(),
                            _image_layout: color_attachment.layout,
                            _resolve_info: color_resolve_attachment.as_ref().map(
                                |color_resolve_attachment| RenderPassStateAttachmentResolveInfo {
                                    _image_view: fb_attachments
                                        [color_resolve_attachment.attachment as usize]
                                        .clone(),
                                    _image_layout: color_resolve_attachment.layout,
                                },
                            ),
                        }
                    })
                })
                .collect(),
            depth_attachment: (subpass_desc.depth_stencil_attachment.as_ref())
                .filter(|depth_stencil_attachment| {
                    (rp_attachments[depth_stencil_attachment.attachment as usize]
                        .format
                        .unwrap())
                    .aspects()
                    .intersects(ImageAspects::DEPTH)
                })
                .map(|depth_stencil_attachment| RenderPassStateAttachmentInfo {
                    image_view: fb_attachments[depth_stencil_attachment.attachment as usize]
                        .clone(),
                    _image_layout: depth_stencil_attachment.layout,
                    _resolve_info: subpass_desc.depth_stencil_resolve_attachment.as_ref().map(
                        |depth_stencil_resolve_attachment| RenderPassStateAttachmentResolveInfo {
                            _image_view: fb_attachments
                                [depth_stencil_resolve_attachment.attachment as usize]
                                .clone(),
                            _image_layout: depth_stencil_resolve_attachment.layout,
                        },
                    ),
                }),
            stencil_attachment: (subpass_desc.depth_stencil_attachment.as_ref())
                .filter(|depth_stencil_attachment| {
                    (rp_attachments[depth_stencil_attachment.attachment as usize]
                        .format
                        .unwrap())
                    .aspects()
                    .intersects(ImageAspects::STENCIL)
                })
                .map(|depth_stencil_attachment| RenderPassStateAttachmentInfo {
                    image_view: fb_attachments[depth_stencil_attachment.attachment as usize]
                        .clone(),
                    _image_layout: depth_stencil_attachment
                        .stencil_layout
                        .unwrap_or(depth_stencil_attachment.layout),
                    _resolve_info: subpass_desc.depth_stencil_resolve_attachment.as_ref().map(
                        |depth_stencil_resolve_attachment| RenderPassStateAttachmentResolveInfo {
                            _image_view: fb_attachments
                                [depth_stencil_resolve_attachment.attachment as usize]
                                .clone(),
                            _image_layout: depth_stencil_resolve_attachment
                                .stencil_layout
                                .unwrap_or(depth_stencil_resolve_attachment.layout),
                        },
                    ),
                }),
        }
    }

    pub(in crate::command_buffer) fn from_rendering_info(info: &RenderingInfo) -> Self {
        Self {
            color_attachments: (info.color_attachments.iter())
                .map(|atch_info| {
                    (atch_info.as_ref()).map(|atch_info| RenderPassStateAttachmentInfo {
                        image_view: atch_info.image_view.clone(),
                        _image_layout: atch_info.image_layout,
                        _resolve_info: atch_info.resolve_info.as_ref().map(|resolve_atch_info| {
                            RenderPassStateAttachmentResolveInfo {
                                _image_view: resolve_atch_info.image_view.clone(),
                                _image_layout: resolve_atch_info.image_layout,
                            }
                        }),
                    })
                })
                .collect(),
            depth_attachment: (info.depth_attachment.as_ref()).map(|atch_info| {
                RenderPassStateAttachmentInfo {
                    image_view: atch_info.image_view.clone(),
                    _image_layout: atch_info.image_layout,
                    _resolve_info: atch_info.resolve_info.as_ref().map(|resolve_atch_info| {
                        RenderPassStateAttachmentResolveInfo {
                            _image_view: resolve_atch_info.image_view.clone(),
                            _image_layout: resolve_atch_info.image_layout,
                        }
                    }),
                }
            }),
            stencil_attachment: (info.stencil_attachment.as_ref()).map(|atch_info| {
                RenderPassStateAttachmentInfo {
                    image_view: atch_info.image_view.clone(),
                    _image_layout: atch_info.image_layout,
                    _resolve_info: atch_info.resolve_info.as_ref().map(|resolve_atch_info| {
                        RenderPassStateAttachmentResolveInfo {
                            _image_view: resolve_atch_info.image_view.clone(),
                            _image_layout: resolve_atch_info.image_layout,
                        }
                    }),
                }
            }),
        }
    }
}

pub(in crate::command_buffer) struct RenderPassStateAttachmentInfo {
    pub(in crate::command_buffer) image_view: Arc<dyn ImageViewAbstract>,
    pub(in crate::command_buffer) _image_layout: ImageLayout,
    pub(in crate::command_buffer) _resolve_info: Option<RenderPassStateAttachmentResolveInfo>,
}

pub(in crate::command_buffer) struct RenderPassStateAttachmentResolveInfo {
    pub(in crate::command_buffer) _image_view: Arc<dyn ImageViewAbstract>,
    pub(in crate::command_buffer) _image_layout: ImageLayout,
}

pub(in crate::command_buffer) struct DescriptorSetState {
    pub(in crate::command_buffer) descriptor_sets: HashMap<u32, SetOrPush>,
    pub(in crate::command_buffer) pipeline_layout: Arc<PipelineLayout>,
}

#[derive(Clone)]
pub(in crate::command_buffer) enum SetOrPush {
    Set(DescriptorSetWithOffsets),
    Push(DescriptorSetResources),
}

impl SetOrPush {
    pub(in crate::command_buffer) fn resources(&self) -> &DescriptorSetResources {
        match self {
            Self::Set(set) => set.as_ref().0.resources(),
            Self::Push(resources) => resources,
        }
    }

    #[inline]
    pub(in crate::command_buffer) fn dynamic_offsets(&self) -> &[u32] {
        match self {
            Self::Set(set) => set.as_ref().1,
            Self::Push(_) => &[],
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::command_buffer) struct StencilStateDynamic {
    pub(in crate::command_buffer) front: Option<u32>,
    pub(in crate::command_buffer) back: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::command_buffer) struct StencilOpStateDynamic {
    pub(in crate::command_buffer) front: Option<StencilOps>,
    pub(in crate::command_buffer) back: Option<StencilOps>,
}

pub(in crate::command_buffer) struct QueryState {
    pub(in crate::command_buffer) query_pool: Arc<QueryPool>,
    pub(in crate::command_buffer) query: u32,
    pub(in crate::command_buffer) flags: QueryControlFlags,
    pub(in crate::command_buffer) in_subpass: bool,
}
