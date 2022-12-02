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
    buffer::BufferAccess,
    command_buffer::{
        allocator::{
            CommandBufferAllocator, CommandBufferBuilderAlloc, StandardCommandBufferAllocator,
        },
        sys::CommandBufferBeginInfo,
        BuildError, CommandBufferBeginError, CommandBufferInheritanceInfo,
        CommandBufferInheritanceRenderPassInfo, CommandBufferInheritanceRenderPassType,
        CommandBufferInheritanceRenderingInfo, CommandBufferLevel, CommandBufferUsage,
        SubpassContents,
    },
    descriptor_set::{DescriptorSetResources, DescriptorSetWithOffsets},
    device::{Device, DeviceOwned, QueueFamilyProperties},
    format::{Format, FormatFeatures},
    image::ImageAspects,
    pipeline::{
        graphics::{
            color_blend::LogicOp,
            depth_stencil::{CompareOp, StencilOps},
            input_assembly::{IndexType, PrimitiveTopology},
            rasterization::{CullMode, DepthBias, FrontFace, LineStipple},
            viewport::{Scissor, Viewport},
        },
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    },
    query::{QueryControlFlags, QueryType},
    range_set::RangeSet,
    render_pass::{Framebuffer, Subpass},
    OomError, RequiresOneOf, VulkanError, VulkanObject,
};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    any::Any,
    collections::{hash_map::Entry, HashMap},
    marker::PhantomData,
    ops::RangeInclusive,
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

    resources: Vec<Box<dyn Any + Send + Sync>>,
    current_state: CurrentState,

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

        let mut current_state: CurrentState = Default::default();

        if let Some(inheritance_info) = &inheritance_info {
            let &CommandBufferInheritanceInfo {
                ref render_pass,
                occlusion_query: _,
                query_statistics_flags: _,
                _ne: _,
            } = inheritance_info;

            if let Some(render_pass) = render_pass {
                current_state.render_pass = Some(RenderPassState::from_inheritance(render_pass));
            }
        }

        Ok(CommandBufferBuilder {
            builder_alloc,
            inheritance_info,
            queue_family_index,
            usage,

            resources: Vec::new(),
            current_state,

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
        if self.current_state.render_pass.is_some() {
            return Err(BuildError::RenderPassActive);
        }

        if !self.current_state.queries.is_empty() {
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
        if !self.current_state.queries.is_empty() {
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
struct CurrentState {
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

impl CurrentState {
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
    render_pass: RenderPassStateType,
    view_mask: u32,
}

impl RenderPassState {
    fn from_inheritance(render_pass: &CommandBufferInheritanceRenderPassType) -> Self {
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

                RenderPassState {
                    contents: SubpassContents::Inline,
                    render_area_offset,
                    render_area_extent,
                    render_pass: BeginRenderPassState {
                        subpass: info.subpass.clone(),
                        framebuffer: info.framebuffer.clone(),
                    }
                    .into(),
                    view_mask: info.subpass.subpass_desc().view_mask,
                }
            }
            CommandBufferInheritanceRenderPassType::BeginRendering(info) => RenderPassState {
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
    attachments: Option<BeginRenderingAttachments>,
    color_attachment_formats: Vec<Option<Format>>,
    depth_attachment_format: Option<Format>,
    stencil_attachment_format: Option<Format>,
    pipeline_used: bool,
}

struct BeginRenderingAttachments {
    color_attachments: Vec<Option<RenderingAttachmentInfo>>,
    depth_attachment: Option<RenderingAttachmentInfo>,
    stencil_attachment: Option<RenderingAttachmentInfo>,
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
