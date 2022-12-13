// Copyright (c) 2022 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub use self::{
    bind_push::*, clear::*, copy::*, debug::*, dynamic_state::*, pipeline::*, query::*,
    render_pass::*, secondary::*, sync::*,
};
use super::{PrimaryCommandBuffer, SecondaryCommandBuffer, SubmitState};
pub use crate::command_buffer::{
    BlitImageInfo, BufferCopy, BufferImageCopy, ClearAttachment, ClearColorImageInfo,
    ClearDepthStencilImageInfo, ClearError, ClearRect, CopyBufferInfo, CopyBufferInfoTyped,
    CopyBufferToImageInfo, CopyError, CopyErrorResource, CopyImageInfo, CopyImageToBufferInfo,
    DebugUtilsError, ExecuteCommandsError, FillBufferInfo, ImageBlit, ImageCopy, ImageResolve,
    PipelineExecutionError, QueryError, RenderPassBeginInfo, RenderPassError,
    RenderingAttachmentInfo, RenderingAttachmentResolveInfo, RenderingInfo, ResolveImageInfo,
};
use crate::{
    buffer::{sys::Buffer, BufferAccess},
    command_buffer::{
        allocator::{
            CommandBufferAllocator, CommandBufferBuilderAlloc, StandardCommandBufferAllocator,
        },
        sys::CommandBufferBeginInfo,
        BuildError, CommandBufferBeginError, CommandBufferInheritanceInfo,
        CommandBufferInheritanceRenderPassInfo, CommandBufferInheritanceRenderPassType,
        CommandBufferInheritanceRenderingInfo, CommandBufferLevel, CommandBufferUsage,
        ResourceInCommand, ResourceUseRef, SubpassContents,
    },
    descriptor_set::{DescriptorSetResources, DescriptorSetWithOffsets},
    device::{Device, DeviceOwned, QueueFamilyProperties, QueueFlags},
    format::FormatFeatures,
    image::{sys::Image, ImageAspects, ImageLayout, ImageSubresourceRange, ImageViewAbstract},
    pipeline::{
        graphics::{
            color_blend::LogicOp,
            depth_stencil::{CompareOp, StencilOps},
            input_assembly::{IndexType, PrimitiveTopology},
            rasterization::{CullMode, DepthBias, FrontFace, LineStipple},
            render_pass::PipelineRenderingCreateInfo,
            viewport::{Scissor, Viewport},
        },
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    },
    query::{QueryControlFlags, QueryType},
    range_map::RangeMap,
    range_set::RangeSet,
    render_pass::{Framebuffer, LoadOp, StoreOp, Subpass},
    sync::{
        BufferMemoryBarrier, DependencyInfo, ImageMemoryBarrier, PipelineStage,
        PipelineStageAccess, PipelineStageAccessSet, PipelineStages,
    },
    DeviceSize, OomError, RequiresOneOf, VulkanError, VulkanObject,
};
use ahash::HashMap;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    any::Any,
    collections::hash_map::Entry,
    marker::PhantomData,
    ops::{Range, RangeInclusive},
    ptr,
    sync::{atomic::AtomicBool, Arc},
};

mod bind_push;
mod clear;
mod copy;
mod debug;
mod dynamic_state;
mod pipeline;
mod query;
mod render_pass;
mod secondary;
mod sync;

/// Records commands to a command buffer.
pub struct CommandBufferBuilder<L, A = StandardCommandBufferAllocator>
where
    A: CommandBufferAllocator,
{
    builder_alloc: A::Builder,
    inheritance_info: Option<CommandBufferInheritanceInfo>, // Must be `None` in a primary command buffer and `Some` in a secondary command buffer.
    queue_family_index: u32,
    usage: CommandBufferUsage,

    next_command_index: usize,
    resources: Vec<Box<dyn Any + Send + Sync>>,
    builder_state: CommandBufferBuilderState,
    resources_usage_state: ResourcesState,

    _data: PhantomData<L>,
}

unsafe impl<L, A> DeviceOwned for CommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.builder_alloc.device()
    }
}

unsafe impl<L, A> VulkanObject for CommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.builder_alloc.inner().handle()
    }
}

impl<A> CommandBufferBuilder<PrimaryCommandBuffer, A>
where
    A: CommandBufferAllocator,
{
    /// Starts recording a primary command buffer.
    #[inline]
    pub fn primary(
        allocator: &A,
        queue_family_index: u32,
        usage: CommandBufferUsage,
    ) -> Result<Self, CommandBufferBeginError> {
        unsafe {
            CommandBufferBuilder::begin(
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

impl<A> CommandBufferBuilder<SecondaryCommandBuffer, A>
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
    ) -> Result<Self, CommandBufferBeginError> {
        unsafe {
            CommandBufferBuilder::begin(
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

impl<L, A> CommandBufferBuilder<L, A>
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
    ) -> Result<Self, CommandBufferBeginError> {
        Self::validate_begin(allocator.device(), queue_family_index, level, &begin_info)?;
        Ok(Self::begin_unchecked(
            allocator,
            queue_family_index,
            level,
            begin_info,
        )?)
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

    unsafe fn begin_unchecked(
        allocator: &A,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<Self, OomError> {
        let CommandBufferBeginInfo {
            usage,
            inheritance_info,
            _ne: _,
        } = begin_info;

        let builder_alloc = allocator
            .allocate(queue_family_index, level, 1)?
            .next()
            .expect("requested one command buffer from the command pool, but got zero");

        {
            let device = builder_alloc.device();

            let mut flags = ash::vk::CommandBufferUsageFlags::from(usage);
            let mut inheritance_info_vk = None;
            let mut inheritance_rendering_info_vk = None;
            let mut color_attachment_formats_vk: SmallVec<[_; 4]> = SmallVec::new();

            if let Some(inheritance_info) = &inheritance_info {
                let &CommandBufferInheritanceInfo {
                    ref render_pass,
                    occlusion_query,
                    query_statistics_flags,
                    _ne: _,
                } = inheritance_info;

                let inheritance_info_vk =
                    inheritance_info_vk.insert(ash::vk::CommandBufferInheritanceInfo {
                        render_pass: ash::vk::RenderPass::null(),
                        subpass: 0,
                        framebuffer: ash::vk::Framebuffer::null(),
                        occlusion_query_enable: ash::vk::FALSE,
                        query_flags: ash::vk::QueryControlFlags::empty(),
                        pipeline_statistics: query_statistics_flags.into(),
                        ..Default::default()
                    });

                if let Some(flags) = occlusion_query {
                    inheritance_info_vk.occlusion_query_enable = ash::vk::TRUE;

                    if flags.intersects(QueryControlFlags::PRECISE) {
                        inheritance_info_vk.query_flags = ash::vk::QueryControlFlags::PRECISE;
                    }
                }

                if let Some(render_pass) = render_pass {
                    flags |= ash::vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE;

                    match render_pass {
                        CommandBufferInheritanceRenderPassType::BeginRenderPass(
                            render_pass_info,
                        ) => {
                            let &CommandBufferInheritanceRenderPassInfo {
                                ref subpass,
                                ref framebuffer,
                            } = render_pass_info;

                            inheritance_info_vk.render_pass = subpass.render_pass().handle();
                            inheritance_info_vk.subpass = subpass.index();
                            inheritance_info_vk.framebuffer = framebuffer
                                .as_ref()
                                .map(|fb| fb.handle())
                                .unwrap_or_default();
                        }
                        CommandBufferInheritanceRenderPassType::BeginRendering(rendering_info) => {
                            let &CommandBufferInheritanceRenderingInfo {
                                view_mask,
                                ref color_attachment_formats,
                                depth_attachment_format,
                                stencil_attachment_format,
                                rasterization_samples,
                            } = rendering_info;

                            color_attachment_formats_vk.extend(
                                color_attachment_formats.iter().map(|format| {
                                    format.map_or(ash::vk::Format::UNDEFINED, Into::into)
                                }),
                            );

                            let inheritance_rendering_info_vk = inheritance_rendering_info_vk
                                .insert(ash::vk::CommandBufferInheritanceRenderingInfo {
                                    flags: ash::vk::RenderingFlags::empty(),
                                    view_mask,
                                    color_attachment_count: color_attachment_formats_vk.len()
                                        as u32,
                                    p_color_attachment_formats: color_attachment_formats_vk
                                        .as_ptr(),
                                    depth_attachment_format: depth_attachment_format
                                        .map_or(ash::vk::Format::UNDEFINED, Into::into),
                                    stencil_attachment_format: stencil_attachment_format
                                        .map_or(ash::vk::Format::UNDEFINED, Into::into),
                                    rasterization_samples: rasterization_samples.into(),
                                    ..Default::default()
                                });

                            inheritance_info_vk.p_next =
                                inheritance_rendering_info_vk as *const _ as *const _;
                        }
                    }
                }
            }

            let begin_info_vk = ash::vk::CommandBufferBeginInfo {
                flags,
                p_inheritance_info: inheritance_info_vk
                    .as_ref()
                    .map_or(ptr::null(), |info| info),
                ..Default::default()
            };

            let fns = device.fns();

            (fns.v1_0.begin_command_buffer)(builder_alloc.inner().handle(), &begin_info_vk)
                .result()
                .map_err(VulkanError::from)?;
        }

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

        Ok(CommandBufferBuilder {
            builder_alloc,
            inheritance_info,
            queue_family_index,
            usage,

            next_command_index: 0,
            resources: Vec::new(),
            builder_state,
            resources_usage_state: Default::default(),

            _data: PhantomData,
        })
    }

    fn queue_family_properties(&self) -> &QueueFamilyProperties {
        &self.device().physical_device().queue_family_properties()[self.queue_family_index as usize]
    }
}

impl<A> CommandBufferBuilder<PrimaryCommandBuffer<A::Alloc>, A>
where
    A: CommandBufferAllocator,
{
    /// Builds the command buffer.
    pub fn build(self) -> Result<PrimaryCommandBuffer<A::Alloc>, BuildError> {
        if self.builder_state.render_pass.is_some() {
            return Err(BuildError::RenderPassActive);
        }

        if !self.builder_state.queries.is_empty() {
            return Err(BuildError::QueryActive);
        }

        Ok(unsafe { self.build_unchecked()? })
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn build_unchecked(self) -> Result<PrimaryCommandBuffer<A::Alloc>, OomError> {
        let fns = self.device().fns();
        (fns.v1_0.end_command_buffer)(self.builder_alloc.inner().handle())
            .result()
            .map_err(VulkanError::from)?;

        Ok(PrimaryCommandBuffer {
            alloc: self.builder_alloc.into_alloc(),
            _usage: self.usage,
            _resources: self.resources,

            _state: Mutex::new(Default::default()),
        })
    }
}

impl<A> CommandBufferBuilder<SecondaryCommandBuffer<A::Alloc>, A>
where
    A: CommandBufferAllocator,
{
    /// Builds the command buffer.
    pub fn build(self) -> Result<SecondaryCommandBuffer<A::Alloc>, BuildError> {
        if !self.builder_state.queries.is_empty() {
            return Err(BuildError::QueryActive);
        }

        Ok(unsafe { self.build_unchecked()? })
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn build_unchecked(self) -> Result<SecondaryCommandBuffer<A::Alloc>, OomError> {
        let fns = self.device().fns();
        (fns.v1_0.end_command_buffer)(self.builder_alloc.inner().handle())
            .result()
            .map_err(VulkanError::from)?;

        let submit_state = match self.usage {
            CommandBufferUsage::MultipleSubmit => SubmitState::ExclusiveUse {
                in_use: AtomicBool::new(false),
            },
            CommandBufferUsage::SimultaneousUse => SubmitState::Concurrent,
            CommandBufferUsage::OneTimeSubmit => SubmitState::OneTime {
                already_submitted: AtomicBool::new(false),
            },
        };

        Ok(SecondaryCommandBuffer {
            alloc: self.builder_alloc.into_alloc(),
            inheritance_info: self.inheritance_info.unwrap(),
            usage: self.usage,

            _resources: self.resources,

            submit_state,
        })
    }
}

/// Holds the current binding and setting state.
#[derive(Default)]
struct CommandBufferBuilderState {
    // Render pass
    render_pass: Option<RenderPassState>,

    // Bind/push
    descriptor_sets: HashMap<PipelineBindPoint, DescriptorSetState>,
    index_buffer: Option<(Arc<dyn BufferAccess>, IndexType)>,
    pipeline_compute: Option<Arc<ComputePipeline>>,
    pipeline_graphics: Option<Arc<GraphicsPipeline>>,
    vertex_buffers: HashMap<u32, Arc<dyn BufferAccess>>,
    push_constants: RangeSet<u32>,
    push_constants_pipeline_layout: Option<Arc<PipelineLayout>>,

    // Dynamic state
    blend_constants: Option<[f32; 4]>,
    color_write_enable: Option<SmallVec<[bool; 4]>>,
    cull_mode: Option<CullMode>,
    depth_bias: Option<DepthBias>,
    depth_bias_enable: Option<bool>,
    depth_bounds: Option<RangeInclusive<f32>>,
    depth_bounds_test_enable: Option<bool>,
    depth_compare_op: Option<CompareOp>,
    depth_test_enable: Option<bool>,
    depth_write_enable: Option<bool>,
    discard_rectangle: HashMap<u32, Scissor>,
    front_face: Option<FrontFace>,
    line_stipple: Option<LineStipple>,
    line_width: Option<f32>,
    logic_op: Option<LogicOp>,
    patch_control_points: Option<u32>,
    primitive_restart_enable: Option<bool>,
    primitive_topology: Option<PrimitiveTopology>,
    rasterizer_discard_enable: Option<bool>,
    scissor: HashMap<u32, Scissor>,
    scissor_with_count: Option<SmallVec<[Scissor; 2]>>,
    stencil_compare_mask: StencilStateDynamic,
    stencil_op: StencilOpStateDynamic,
    stencil_reference: StencilStateDynamic,
    stencil_test_enable: Option<bool>,
    stencil_write_mask: StencilStateDynamic,
    viewport: HashMap<u32, Viewport>,
    viewport_with_count: Option<SmallVec<[Viewport; 2]>>,

    // Active queries
    queries: HashMap<ash::vk::QueryType, QueryState>,
}

impl CommandBufferBuilderState {
    fn reset_dynamic_states(&mut self, states: impl IntoIterator<Item = DynamicState>) {
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

    fn invalidate_descriptor_sets(
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

struct RenderPassState {
    contents: SubpassContents,
    render_area_offset: [u32; 2],
    render_area_extent: [u32; 2],

    rendering_info: PipelineRenderingCreateInfo,
    attachments: Option<RenderPassStateAttachments>,

    render_pass: RenderPassStateType,
}

impl RenderPassState {
    fn from_inheritance(render_pass: &CommandBufferInheritanceRenderPassType) -> Self {
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

enum RenderPassStateType {
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

struct BeginRenderPassState {
    subpass: Subpass,
    framebuffer: Option<Arc<Framebuffer>>,
}

struct BeginRenderingState {
    pipeline_used: bool,
}

struct RenderPassStateAttachments {
    color_attachments: Vec<Option<RenderPassStateAttachmentInfo>>,
    depth_attachment: Option<RenderPassStateAttachmentInfo>,
    stencil_attachment: Option<RenderPassStateAttachmentInfo>,
}

impl RenderPassStateAttachments {
    fn from_subpass(subpass: &Subpass, framebuffer: &Framebuffer) -> Self {
        let subpass_desc = subpass.subpass_desc();
        let rp_attachments = subpass.render_pass().attachments();
        let fb_attachments = framebuffer.attachments();

        Self {
            color_attachments: (subpass_desc.color_attachments.iter().enumerate())
                .map(|(index, atch_ref)| {
                    (atch_ref.as_ref()).map(|atch_ref| RenderPassStateAttachmentInfo {
                        image_view: fb_attachments[atch_ref.attachment as usize].clone(),
                        image_layout: atch_ref.layout,
                        load_access: subpass
                            .load_op(atch_ref.attachment)
                            .and_then(color_load_access),
                        store_access: subpass
                            .store_op(atch_ref.attachment)
                            .and_then(color_store_access),
                        resolve_info: (subpass_desc.resolve_attachments.get(index))
                            .and_then(|atch_ref| atch_ref.as_ref())
                            .map(|atch_ref| RenderPassStateAttachmentResolveInfo {
                                image_view: fb_attachments[atch_ref.attachment as usize].clone(),
                                image_layout: atch_ref.layout,
                                load_access: subpass
                                    .load_op(atch_ref.attachment)
                                    .and_then(color_load_access),
                                store_access: subpass
                                    .store_op(atch_ref.attachment)
                                    .and_then(color_store_access),
                            }),
                    })
                })
                .collect(),
            depth_attachment: (subpass_desc.depth_stencil_attachment.as_ref())
                .filter(|atch_ref| {
                    (rp_attachments[atch_ref.attachment as usize].format.unwrap())
                        .aspects()
                        .intersects(ImageAspects::DEPTH)
                })
                .map(|atch_ref| RenderPassStateAttachmentInfo {
                    image_view: fb_attachments[atch_ref.attachment as usize].clone(),
                    image_layout: atch_ref.layout,
                    load_access: subpass
                        .load_op(atch_ref.attachment)
                        .and_then(depth_stencil_load_access),
                    store_access: subpass
                        .store_op(atch_ref.attachment)
                        .and_then(depth_stencil_store_access),
                    resolve_info: None,
                }),
            stencil_attachment: (subpass_desc.depth_stencil_attachment.as_ref())
                .filter(|atch_ref| {
                    (rp_attachments[atch_ref.attachment as usize].format.unwrap())
                        .aspects()
                        .intersects(ImageAspects::STENCIL)
                })
                .map(|atch_ref| RenderPassStateAttachmentInfo {
                    image_view: fb_attachments[atch_ref.attachment as usize].clone(),
                    image_layout: atch_ref.layout,
                    load_access: subpass
                        .stencil_load_op(atch_ref.attachment)
                        .and_then(depth_stencil_load_access),
                    store_access: subpass
                        .stencil_store_op(atch_ref.attachment)
                        .and_then(depth_stencil_store_access),
                    resolve_info: None,
                }),
        }
    }

    fn from_rendering_info(info: &RenderingInfo) -> Self {
        Self {
            color_attachments: (info.color_attachments.iter())
                .map(|atch_info| {
                    (atch_info.as_ref()).map(|atch_info| RenderPassStateAttachmentInfo {
                        image_view: atch_info.image_view.clone(),
                        image_layout: atch_info.image_layout,
                        load_access: color_load_access(atch_info.load_op),
                        store_access: color_store_access(atch_info.store_op),
                        resolve_info: atch_info.resolve_info.as_ref().map(|resolve_atch_info| {
                            RenderPassStateAttachmentResolveInfo {
                                image_view: resolve_atch_info.image_view.clone(),
                                image_layout: resolve_atch_info.image_layout,
                                load_access: None,
                                store_access: None,
                            }
                        }),
                    })
                })
                .collect(),
            depth_attachment: (info.depth_attachment.as_ref()).map(|atch_info| {
                RenderPassStateAttachmentInfo {
                    image_view: atch_info.image_view.clone(),
                    image_layout: atch_info.image_layout,
                    load_access: depth_stencil_load_access(atch_info.load_op),
                    store_access: depth_stencil_store_access(atch_info.store_op),
                    resolve_info: atch_info.resolve_info.as_ref().map(|resolve_atch_info| {
                        RenderPassStateAttachmentResolveInfo {
                            image_view: resolve_atch_info.image_view.clone(),
                            image_layout: resolve_atch_info.image_layout,
                            load_access: None,
                            store_access: None,
                        }
                    }),
                }
            }),
            stencil_attachment: (info.stencil_attachment.as_ref()).map(|atch_info| {
                RenderPassStateAttachmentInfo {
                    image_view: atch_info.image_view.clone(),
                    image_layout: atch_info.image_layout,
                    load_access: depth_stencil_load_access(atch_info.load_op),
                    store_access: depth_stencil_store_access(atch_info.store_op),
                    resolve_info: atch_info.resolve_info.as_ref().map(|resolve_atch_info| {
                        RenderPassStateAttachmentResolveInfo {
                            image_view: resolve_atch_info.image_view.clone(),
                            image_layout: resolve_atch_info.image_layout,
                            load_access: None,
                            store_access: None,
                        }
                    }),
                }
            }),
        }
    }
}

fn color_load_access(load_op: LoadOp) -> Option<PipelineStageAccess> {
    match load_op {
        LoadOp::Load => Some(PipelineStageAccess::ColorAttachmentOutput_ColorAttachmentRead),
        LoadOp::Clear => Some(PipelineStageAccess::ColorAttachmentOutput_ColorAttachmentWrite),
        LoadOp::DontCare => Some(PipelineStageAccess::ColorAttachmentOutput_ColorAttachmentWrite),
        //LoadOp::None => None,
    }
}

fn depth_stencil_load_access(load_op: LoadOp) -> Option<PipelineStageAccess> {
    match load_op {
        LoadOp::Load => Some(PipelineStageAccess::EarlyFragmentTests_DepthStencilAttachmentRead),
        LoadOp::Clear => Some(PipelineStageAccess::EarlyFragmentTests_DepthStencilAttachmentWrite),
        LoadOp::DontCare => {
            Some(PipelineStageAccess::EarlyFragmentTests_DepthStencilAttachmentWrite)
        } //LoadOp::None => None,
    }
}

fn color_store_access(store_op: StoreOp) -> Option<PipelineStageAccess> {
    match store_op {
        StoreOp::Store => Some(PipelineStageAccess::ColorAttachmentOutput_ColorAttachmentWrite),
        StoreOp::DontCare => Some(PipelineStageAccess::ColorAttachmentOutput_ColorAttachmentWrite),
        // StoreOp::None => None,
    }
}

fn depth_stencil_store_access(store_op: StoreOp) -> Option<PipelineStageAccess> {
    match store_op {
        StoreOp::Store => Some(PipelineStageAccess::LateFragmentTests_DepthStencilAttachmentWrite),
        StoreOp::DontCare => {
            Some(PipelineStageAccess::LateFragmentTests_DepthStencilAttachmentWrite)
        } // StoreOp::None => None,
    }
}

struct RenderPassStateAttachmentInfo {
    image_view: Arc<dyn ImageViewAbstract>,
    image_layout: ImageLayout,
    load_access: Option<PipelineStageAccess>,
    store_access: Option<PipelineStageAccess>,
    resolve_info: Option<RenderPassStateAttachmentResolveInfo>,
}

struct RenderPassStateAttachmentResolveInfo {
    image_view: Arc<dyn ImageViewAbstract>,
    image_layout: ImageLayout,
    load_access: Option<PipelineStageAccess>,
    store_access: Option<PipelineStageAccess>,
}

struct DescriptorSetState {
    descriptor_sets: HashMap<u32, SetOrPush>,
    pipeline_layout: Arc<PipelineLayout>,
}

#[derive(Clone)]
enum SetOrPush {
    Set(DescriptorSetWithOffsets),
    Push(DescriptorSetResources),
}

impl SetOrPush {
    pub fn resources(&self) -> &DescriptorSetResources {
        match self {
            Self::Set(set) => set.as_ref().0.resources(),
            Self::Push(resources) => resources,
        }
    }

    #[inline]
    pub fn dynamic_offsets(&self) -> &[u32] {
        match self {
            Self::Set(set) => set.as_ref().1,
            Self::Push(_) => &[],
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct StencilStateDynamic {
    front: Option<u32>,
    back: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct StencilOpStateDynamic {
    front: Option<StencilOps>,
    back: Option<StencilOps>,
}

struct QueryState {
    query_pool: ash::vk::QueryPool,
    query: u32,
    ty: QueryType,
    flags: QueryControlFlags,
    in_subpass: bool,
}

#[derive(Debug, Default)]
struct ResourcesState {
    buffers: HashMap<Arc<Buffer>, RangeMap<DeviceSize, BufferRangeState>>,
    images: HashMap<Arc<Image>, RangeMap<DeviceSize, ImageRangeState>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BufferRangeState {
    resource_uses: Vec<ResourceUseRef>,
    memory_access: MemoryAccessState,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ImageRangeState {
    resource_uses: Vec<ResourceUseRef>,
    memory_access: MemoryAccessState,
    expected_layout: ImageLayout,
    current_layout: ImageLayout,
}

impl ResourcesState {
    fn record_buffer_access(
        &mut self,
        use_ref: &ResourceUseRef,
        buffer: &Arc<Buffer>,
        range: Range<DeviceSize>,
        stage_access: PipelineStageAccess,
    ) {
        let range_map = self.buffers.entry(buffer.clone()).or_insert_with(|| {
            [(0..buffer.size(), Default::default())]
                .into_iter()
                .collect()
        });
        range_map.split_at(&range.start);
        range_map.split_at(&range.end);

        for (_range, state) in range_map.range_mut(&range) {
            state.resource_uses.push(*use_ref);
            state.memory_access.record_access(use_ref, stage_access);
        }
    }

    fn record_image_access(
        &mut self,
        use_ref: &ResourceUseRef,
        image: &Arc<Image>,
        subresource_range: ImageSubresourceRange,
        stage_access: PipelineStageAccess,
        image_layout: ImageLayout,
    ) {
        let range_map = self.images.entry(image.clone()).or_insert_with(|| {
            [(0..image.range_size(), Default::default())]
                .into_iter()
                .collect()
        });

        for range in image.iter_ranges(subresource_range) {
            range_map.split_at(&range.start);
            range_map.split_at(&range.end);

            for (_range, state) in range_map.range_mut(&range) {
                if state.resource_uses.is_empty() {
                    state.expected_layout = image_layout;
                }

                state.resource_uses.push(*use_ref);
                state.memory_access.record_access(use_ref, stage_access);
            }
        }
    }

    fn record_pipeline_barrier(
        &mut self,
        command_index: usize,
        command_name: &'static str,
        dependency_info: &DependencyInfo,
        queue_flags: QueueFlags,
    ) {
        for barrier in &dependency_info.buffer_memory_barriers {
            let barrier_scopes = BarrierScopes::from_buffer_memory_barrier(barrier, queue_flags);
            let &BufferMemoryBarrier {
                src_stages: _,
                src_access: _,
                dst_stages: _,
                dst_access: _,
                queue_family_ownership_transfer: _,
                ref buffer,
                ref range,
                _ne: _,
            } = barrier;

            let range_map = self.buffers.entry(buffer.clone()).or_insert_with(|| {
                [(0..buffer.size(), Default::default())]
                    .into_iter()
                    .collect()
            });
            range_map.split_at(&range.start);
            range_map.split_at(&range.end);

            for (_range, state) in range_map.range_mut(range) {
                state.memory_access.record_barrier(&barrier_scopes, None);
            }
        }

        for (index, barrier) in dependency_info.image_memory_barriers.iter().enumerate() {
            let index = index as u32;
            let barrier_scopes = BarrierScopes::from_image_memory_barrier(barrier, queue_flags);
            let &ImageMemoryBarrier {
                src_stages: _,
                src_access: _,
                dst_stages: _,
                dst_access: _,
                old_layout,
                new_layout,
                queue_family_ownership_transfer: _,
                ref image,
                ref subresource_range,
                _ne,
            } = barrier;

            // This is only used if there is a layout transition.
            let use_ref = ResourceUseRef {
                command_index,
                command_name,
                resource_in_command: ResourceInCommand::ImageMemoryBarrier { index },
                secondary_use_ref: None,
            };
            let layout_transition = (old_layout != new_layout).then_some(&use_ref);

            let range_map = self.images.entry(image.clone()).or_insert_with(|| {
                [(0..image.range_size(), Default::default())]
                    .into_iter()
                    .collect()
            });

            for range in image.iter_ranges(subresource_range.clone()) {
                range_map.split_at(&range.start);
                range_map.split_at(&range.end);

                for (_range, state) in range_map.range_mut(&range) {
                    if old_layout != new_layout {
                        if state.resource_uses.is_empty() {
                            state.expected_layout = old_layout;
                        }

                        state.resource_uses.push(ResourceUseRef {
                            command_index,
                            command_name,
                            resource_in_command: ResourceInCommand::ImageMemoryBarrier { index },
                            secondary_use_ref: None,
                        });
                        state.current_layout = new_layout;
                    }

                    state
                        .memory_access
                        .record_barrier(&barrier_scopes, layout_transition);
                }
            }
        }

        for barrier in &dependency_info.buffer_memory_barriers {
            let &BufferMemoryBarrier {
                ref buffer,
                ref range,
                ..
            } = barrier;

            let range_map = self.buffers.get_mut(buffer).unwrap();
            for (_range, state) in range_map.range_mut(range) {
                state.memory_access.apply_pending();
            }
        }

        for barrier in &dependency_info.image_memory_barriers {
            let &ImageMemoryBarrier {
                ref image,
                ref subresource_range,
                ..
            } = barrier;

            let range_map = self.images.get_mut(image).unwrap();
            for range in image.iter_ranges(subresource_range.clone()) {
                for (_range, state) in range_map.range_mut(&range) {
                    state.memory_access.apply_pending();
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct MemoryAccessState {
    mutable: bool,
    last_write: Option<WriteState>,
    reads_since_last_write: HashMap<PipelineStage, ReadState>,

    /// Pending changes that have not yet been applied. This is used during barrier recording.
    pending: Option<PendingWriteState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WriteState {
    use_ref: ResourceUseRef,
    access: PipelineStageAccess,

    /// The `dst_stages` and `dst_access` of all barriers that protect against this write.
    barriers_since: PipelineStageAccessSet,

    /// The `dst_stages` of all barriers that form a dependency chain with this write.
    dependency_chain: PipelineStages,

    /// The union of all `barriers_since` of all `reads_since_last_write`.
    read_barriers_since: PipelineStages,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PendingWriteState {
    /// If this is `Some`, then the barrier is treated as a new write,
    /// and the previous `last_write` is discarded.
    /// Otherwise, the values below are added to the existing `last_write`.
    layout_transition: Option<ResourceUseRef>,

    barriers_since: PipelineStageAccessSet,
    dependency_chain: PipelineStages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReadState {
    use_ref: ResourceUseRef,
    access: PipelineStageAccess,

    /// The `dst_stages` of all barriers that protect against this read.
    /// This always includes the stage of `self`.
    barriers_since: PipelineStages,

    /// Stages of reads recorded after this read,
    /// that were in scope of `barriers_since` at the time of recording.
    /// This always includes the stage of `self`.
    barriered_reads_since: PipelineStages,

    /// Pending changes that have not yet been applied. This is used during barrier recording.
    pending: Option<PendingReadState>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PendingReadState {
    barriers_since: PipelineStages,
}

impl MemoryAccessState {
    fn record_access(&mut self, use_ref: &ResourceUseRef, access: PipelineStageAccess) {
        if access.is_write() {
            self.mutable = true;
            self.last_write = Some(WriteState {
                use_ref: *use_ref,
                access,
                barriers_since: Default::default(),
                dependency_chain: Default::default(),
                read_barriers_since: Default::default(),
            });
            self.reads_since_last_write.clear();
        } else {
            let pipeline_stage = PipelineStage::try_from(access).unwrap();
            let pipeline_stages = PipelineStages::from(pipeline_stage);

            for read_state in self.reads_since_last_write.values_mut() {
                if read_state.barriers_since.intersects(pipeline_stages) {
                    read_state.barriered_reads_since |= pipeline_stages;
                } else {
                    read_state.barriered_reads_since -= pipeline_stages;
                }
            }

            self.reads_since_last_write.insert(
                pipeline_stage,
                ReadState {
                    use_ref: *use_ref,
                    access,
                    barriers_since: pipeline_stages,
                    barriered_reads_since: pipeline_stages,
                    pending: None,
                },
            );
        }
    }

    fn record_barrier(
        &mut self,
        barrier_scopes: &BarrierScopes,
        layout_transition: Option<&ResourceUseRef>,
    ) {
        let skip_reads = if let Some(use_ref) = layout_transition {
            let pending = self.pending.get_or_insert_with(Default::default);
            pending.layout_transition = Some(*use_ref);
            true
        } else {
            self.pending
                .map_or(false, |pending| pending.layout_transition.is_some())
        };

        // If the last write is in the src scope of the barrier, then add the dst scopes.
        // If the barrier includes a layout transition, then that layout transition is
        // considered the last write, and it is always in the src scope of the barrier.
        if layout_transition.is_some()
            || self.last_write.as_ref().map_or(false, |write_state| {
                barrier_scopes
                    .src_access_scope
                    .contains_enum(write_state.access)
                    || barrier_scopes
                        .src_exec_scope
                        .intersects(write_state.dependency_chain)
            })
        {
            let pending = self.pending.get_or_insert_with(Default::default);
            pending.barriers_since |= barrier_scopes.dst_access_scope;
            pending.dependency_chain |= barrier_scopes.dst_exec_scope;
        }

        // A layout transition counts as a write, which means that `reads_since_last_write` will
        // be cleared when applying pending operations.
        // Therefore, there is no need to update the reads.
        if !skip_reads {
            // Gather all reads for which `barriers_since` is in the barrier's `src_exec_scope`.
            let reads_in_src_exec_scope = self.reads_since_last_write.iter().fold(
                PipelineStages::empty(),
                |total, (&stage, read_state)| {
                    if barrier_scopes
                        .src_exec_scope
                        .intersects(read_state.barriers_since)
                    {
                        total.union(stage.into())
                    } else {
                        total
                    }
                },
            );

            for read_state in self.reads_since_last_write.values_mut() {
                if reads_in_src_exec_scope.intersects(read_state.barriered_reads_since) {
                    let pending = read_state.pending.get_or_insert_with(Default::default);
                    pending.barriers_since |= barrier_scopes.dst_exec_scope;
                }
            }
        }
    }

    fn apply_pending(&mut self) {
        if let Some(PendingWriteState {
            layout_transition,
            barriers_since,
            dependency_chain,
        }) = self.pending.take()
        {
            // If there is a pending layout transition, it is treated as the new `last_write`.
            if let Some(use_ref) = layout_transition {
                self.mutable = true;
                self.last_write = Some(WriteState {
                    use_ref,
                    access: PipelineStageAccess::ImageLayoutTransition,
                    barriers_since,
                    dependency_chain,
                    read_barriers_since: Default::default(),
                });
                self.reads_since_last_write.clear();
            } else if let Some(write_state) = &mut self.last_write {
                write_state.barriers_since |= barriers_since;
                write_state.dependency_chain |= dependency_chain;
            }
        }

        for read_state in self.reads_since_last_write.values_mut() {
            if let Some(PendingReadState { barriers_since }) = read_state.pending.take() {
                read_state.barriers_since |= barriers_since;

                if let Some(write_state) = &mut self.last_write {
                    write_state.read_barriers_since |= read_state.barriers_since;
                }
            }
        }
    }
}

struct BarrierScopes {
    src_exec_scope: PipelineStages,
    src_access_scope: PipelineStageAccessSet,
    dst_exec_scope: PipelineStages,
    dst_access_scope: PipelineStageAccessSet,
}

impl BarrierScopes {
    fn from_buffer_memory_barrier(barrier: &BufferMemoryBarrier, queue_flags: QueueFlags) -> Self {
        let src_stages_expanded = barrier.src_stages.expand(queue_flags);
        let src_exec_scope = src_stages_expanded.with_earlier();
        let src_access_scope = PipelineStageAccessSet::from(barrier.src_access)
            & PipelineStageAccessSet::from(src_stages_expanded);

        let dst_stages_expanded = barrier.dst_stages.expand(queue_flags);
        let dst_exec_scope = dst_stages_expanded.with_later();
        let dst_access_scope = PipelineStageAccessSet::from(barrier.dst_access)
            & PipelineStageAccessSet::from(dst_stages_expanded);

        Self {
            src_exec_scope,
            src_access_scope,
            dst_exec_scope,
            dst_access_scope,
        }
    }

    fn from_image_memory_barrier(barrier: &ImageMemoryBarrier, queue_flags: QueueFlags) -> Self {
        let src_stages_expanded = barrier.src_stages.expand(queue_flags);
        let src_exec_scope = src_stages_expanded.with_earlier();
        let src_access_scope = PipelineStageAccessSet::from(barrier.src_access)
            & PipelineStageAccessSet::from(src_stages_expanded);

        let dst_stages_expanded = barrier.dst_stages.expand(queue_flags);
        let dst_exec_scope = dst_stages_expanded.with_later();
        let dst_access_scope = PipelineStageAccessSet::from(barrier.dst_access)
            & PipelineStageAccessSet::from(dst_stages_expanded);

        Self {
            src_exec_scope,
            src_access_scope,
            dst_exec_scope,
            dst_access_scope,
        }
    }
}
