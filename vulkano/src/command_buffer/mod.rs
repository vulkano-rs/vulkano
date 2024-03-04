//! Recording commands to execute on the device.
//!
//! With Vulkan, to get the device to perform work, even relatively simple tasks, you must create a
//! command buffer. A command buffer is a list of commands that will executed by the device.
//! You must first record commands to a recording command buffer, then end the recording to turn it
//! into an actual command buffer, and then it can be used. Depending on how a command buffer is
//! created, it can be used only once, or reused many times.
//!
//! # Command pools and allocators
//!
//! Command buffers are allocated from *command pools*. A command pool holds memory that is used to
//! record the sequence of commands in a command buffer. Command pools are not thread-safe, and
//! therefore commands cannot be recorded to a single command buffer from multiple threads at a
//! time.
//!
//! Raw command pools are unsafe to use, so Vulkano uses [command buffer allocators] to manage
//! command buffers and pools, to ensure their memory is used efficiently, and to protect against
//! invalid usage. Vulkano provides the [`StandardCommandBufferAllocator`] for this purpose, but
//! you can also create your own by implementing the [`CommandBufferAllocator`] trait.
//!
//! # Primary and secondary command buffers
//!
//! There are two levels of command buffers:
//!
//! - A primary command buffer can be executed on a queue, and is the main command buffer level. It
//!   cannot be executed within another command buffer.
//! - A secondary command buffer can only be executed within a primary command buffer, not directly
//!   on a queue.
//!
//! Using secondary command buffers, there is slightly more overhead than using primary command
//! buffers alone, but there are also advantages. A single command buffer cannot be recorded
//! from multiple threads at a time, so if you want to divide the recording work among several
//! threads, each thread must record its own command buffer. While it is possible for these to be
//! all primary command buffers, there are limitations: a render pass or query cannot span multiple
//! primary command buffers, while secondary command buffers can [inherit] this state from their
//! parent primary command buffer. Therefore, to have a single render pass or query that is shared
//! across all the command buffers, you must record secondary command buffers.
//!
//! # Recording a command buffer
//!
//! To record a new command buffer, the most direct way is to create a new
//! [`RecordingCommandBuffer`]. You can then call methods on this object to record new commands to
//! the command buffer. When you are done recording, you call [`end`] to finalise the command
//! buffer and turn it into a [`CommandBuffer`].
//!
//! # Submitting a primary command buffer
//!
//! Once a primary command buffer has finished recording, you can submit the [`CommandBuffer`] to a
//! queue. Submitting a command buffer returns an object that implements the [`GpuFuture`] trait
//! and that represents the moment when the execution will end on the GPU.
//!
//! ```
//! use vulkano::command_buffer::{
//!     CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, RecordingCommandBuffer,
//!     SubpassContents,
//! };
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! # let queue: std::sync::Arc<vulkano::device::Queue> = return;
//! # let vertex_buffer: vulkano::buffer::Subbuffer<[u32]> = return;
//! # let render_pass_begin_info: vulkano::command_buffer::RenderPassBeginInfo = return;
//! # let graphics_pipeline: std::sync::Arc<vulkano::pipeline::graphics::GraphicsPipeline> = return;
//! # let command_buffer_allocator: std::sync::Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator> = return;
//! #
//! let mut cb = RecordingCommandBuffer::new(
//!     command_buffer_allocator.clone(),
//!     queue.queue_family_index(),
//!     CommandBufferLevel::Primary,
//!     CommandBufferBeginInfo::default(),
//! )
//! .unwrap();
//!
//! cb.begin_render_pass(render_pass_begin_info, Default::default())
//!     .unwrap()
//!     .bind_pipeline_graphics(graphics_pipeline.clone())
//!     .unwrap()
//!     .bind_vertex_buffers(0, vertex_buffer.clone())
//!     .unwrap();
//!
//! unsafe {
//!     cb.draw(vertex_buffer.len() as u32, 1, 0, 0).unwrap();
//! }
//!
//! cb.end_render_pass(Default::default()).unwrap();
//!
//! let cb = cb.end().unwrap();
//!
//! let future = cb.execute(queue.clone());
//! ```
//!
//! [`StandardCommandBufferAllocator`]: self::allocator::StandardCommandBufferAllocator
//! [`CommandBufferAllocator`]: self::allocator::CommandBufferAllocator
//! [inherit]: CommandBufferInheritanceInfo
//! [`end`]: RecordingCommandBuffer::end
//! [`GpuFuture`]: crate::sync::GpuFuture

#[allow(unused_imports)] // everything is exported for future-proofing
pub use self::commands::{
    acceleration_structure::*, clear::*, copy::*, debug::*, dynamic_state::*, pipeline::*,
    query::*, render_pass::*, secondary::*, sync::*,
};
pub use self::{
    auto::{CommandBuffer, RecordingCommandBuffer},
    sys::CommandBufferBeginInfo,
    traits::{CommandBufferExecError, CommandBufferExecFuture},
};
use crate::{
    buffer::{Buffer, Subbuffer},
    device::{Device, DeviceOwned},
    format::{Format, FormatFeatures},
    image::{Image, ImageAspects, ImageLayout, ImageSubresourceRange, SampleCount},
    macros::vulkan_enum,
    query::{QueryControlFlags, QueryPipelineStatisticFlags},
    range_map::RangeMap,
    render_pass::{Framebuffer, Subpass},
    sync::{
        semaphore::{Semaphore, SemaphoreType},
        PipelineStageAccessFlags, PipelineStages,
    },
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
#[cfg(doc)]
use crate::{
    device::{DeviceFeatures, DeviceProperties},
    pipeline::graphics::vertex_input::VertexInputRate,
};
use ahash::HashMap;
use bytemuck::{Pod, Zeroable};
use std::{ops::Range, sync::Arc};

pub mod allocator;
pub mod auto;
mod commands;
pub mod pool;
pub mod sys;
mod traits;

/// Used as buffer contents to provide input for the
/// [`RecordingCommandBuffer::dispatch_indirect`] command.
///
/// # Safety
///
/// - The `x`, `y` and `z` values must not be greater than the respective elements of the
/// [`max_compute_work_group_count`](DeviceProperties::max_compute_work_group_count) device limit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, PartialEq, Eq)]
pub struct DispatchIndirectCommand {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Used as buffer contents to provide input for the
/// [`RecordingCommandBuffer::draw_indirect`] command.
///
/// # Safety
///
/// - Every vertex number within the specified range must fall within the range of the bound
///   vertex-rate vertex buffers.
/// - Every instance number within the specified range must fall within the range of the bound
///   instance-rate vertex buffers.
/// - If the [`draw_indirect_first_instance`](DeviceFeatures::draw_indirect_first_instance) feature
///   is not enabled, then `first_instance` must be `0`.
/// - If an [instance divisor](VertexInputRate::Instance) other than 1 is used, and the
///   [`supports_non_zero_first_instance`](DeviceProperties::supports_non_zero_first_instance)
///   device property is `false`, then `first_instance` must be `0`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, PartialEq, Eq)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Used as buffer contents to provide input for the
/// [`RecordingCommandBuffer::draw_mesh_tasks_indirect`] command.
///
/// # Safety
///
/// - If the graphics pipeline **does not** include a task shader, then the `group_count_x`,
///   `group_count_y` and `group_count_z` values must not be greater than the respective elements
///   of the [`max_mesh_work_group_count`](DeviceProperties::max_mesh_work_group_count) device
///   limit, and the product of these three values must not be greater than the
///   [`max_mesh_work_group_total_count`](DeviceProperties::max_mesh_work_group_total_count) device
///   limit.
/// - If the graphics pipeline **does** include a task shader, then the `group_count_x`,
///   `group_count_y` and `group_count_z` values must not be greater than the respective elements
///   of the [`max_task_work_group_count`](DeviceProperties::max_task_work_group_count) device
///   limit, and the product of these three values must not be greater than the
///   [`max_task_work_group_total_count`](DeviceProperties::max_task_work_group_total_count) device
///   limit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, PartialEq, Eq)]
pub struct DrawMeshTasksIndirectCommand {
    pub group_count_x: u32,
    pub group_count_y: u32,
    pub group_count_z: u32,
}

/// Used as buffer contents to provide input for the
/// [`RecordingCommandBuffer::draw_indexed_indirect`] command.
///
/// # Safety
///
/// - Every index within the specified range must fall within the range of the bound index buffer.
/// - Every vertex number that is retrieved from the index buffer must fall within the range of the
///   bound vertex-rate vertex buffers.
/// - Every vertex number that is retrieved from the index buffer, if it is not the special
///   primitive restart value, must be no greater than the
///   [`max_draw_indexed_index_value`](DeviceProperties::max_draw_indexed_index_value) device
///   limit.
/// - Every instance number within the specified range must fall within the range of the bound
///   instance-rate vertex buffers.
/// - If the [`draw_indirect_first_instance`](DeviceFeatures::draw_indirect_first_instance) feature
///   is not enabled, then `first_instance` must be `0`.
/// - If an [instance divisor](VertexInputRate::Instance) other than 1 is used, and the
///   [`supports_non_zero_first_instance`](DeviceProperties::supports_non_zero_first_instance)
///   device property is `false`, then `first_instance` must be `0`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, PartialEq, Eq)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: u32,
    pub first_instance: u32,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes what a subpass in a command buffer will contain.
    SubpassContents = SubpassContents(i32);

    /// The subpass will only directly contain commands.
    Inline = INLINE,

    /// The subpass will only contain secondary command buffers invocations.
    SecondaryCommandBuffers = SECONDARY_COMMAND_BUFFERS,
}

impl From<SubpassContents> for ash::vk::RenderingFlags {
    #[inline]
    fn from(val: SubpassContents) -> Self {
        match val {
            SubpassContents::Inline => Self::empty(),
            SubpassContents::SecondaryCommandBuffers => Self::CONTENTS_SECONDARY_COMMAND_BUFFERS,
        }
    }
}

vulkan_enum! {
    /// Determines the kind of command buffer to create.
    CommandBufferLevel = CommandBufferLevel(i32);

    /// Primary command buffers can be executed on a queue, and can call secondary command buffers.
    /// Render passes must begin and end within the same primary command buffer.
    Primary = PRIMARY,

    /// Secondary command buffers cannot be executed on a queue, but can be executed by a primary
    /// command buffer. If created for a render pass, they must fit within a single render subpass.
    Secondary = SECONDARY,
}

/// The context that a secondary command buffer can inherit from the primary command
/// buffer it's executed in.
#[derive(Clone, Debug)]
pub struct CommandBufferInheritanceInfo {
    /// If `Some`, the secondary command buffer is required to be executed within a render pass
    /// instance, and can only call draw operations.
    /// If `None`, it must be executed outside a render pass instance, and can execute dispatch and
    /// transfer operations, but not drawing operations.
    ///
    /// The default value is `None`.
    pub render_pass: Option<CommandBufferInheritanceRenderPassType>,

    /// If `Some`, the secondary command buffer is allowed to be executed within a primary that has
    /// an occlusion query active. The inner `QueryControlFlags` specifies which flags the
    /// active occlusion is allowed to have enabled.
    /// If `None`, the primary command buffer cannot have an occlusion query active when this
    /// secondary command buffer is executed.
    ///
    /// The `inherited_queries` feature must be enabled if this is `Some`.
    ///
    /// The default value is `None`.
    pub occlusion_query: Option<QueryControlFlags>,

    /// Which `PipelineStatistics` queries are allowed to be active on the primary command buffer
    /// when this secondary command buffer is executed.
    ///
    /// If this value is not empty, the [`pipeline_statistics_query`] feature must be enabled on
    /// the device.
    ///
    /// The default value is [`QueryPipelineStatisticFlags::empty()`].
    ///
    /// [`pipeline_statistics_query`]: crate::device::DeviceFeatures::pipeline_statistics_query
    pub pipeline_statistics: QueryPipelineStatisticFlags,

    pub _ne: crate::NonExhaustive,
}

impl Default for CommandBufferInheritanceInfo {
    #[inline]
    fn default() -> Self {
        Self {
            render_pass: None,
            occlusion_query: None,
            pipeline_statistics: QueryPipelineStatisticFlags::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl CommandBufferInheritanceInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref render_pass,
            occlusion_query,
            pipeline_statistics,
            _ne: _,
        } = self;

        if let Some(render_pass) = render_pass {
            // VUID-VkCommandBufferBeginInfo-flags-06000
            // VUID-VkCommandBufferBeginInfo-flags-06002
            // Ensured by the definition of the `CommandBufferInheritanceRenderPassType` enum.

            match render_pass {
                CommandBufferInheritanceRenderPassType::BeginRenderPass(render_pass_info) => {
                    render_pass_info
                        .validate(device)
                        .map_err(|err| err.add_context("render_pass"))?;
                }
                CommandBufferInheritanceRenderPassType::BeginRendering(rendering_info) => {
                    rendering_info
                        .validate(device)
                        .map_err(|err| err.add_context("render_pass"))?;
                }
            }
        }

        if let Some(control_flags) = occlusion_query {
            control_flags.validate_device(device).map_err(|err| {
                err.add_context("occlusion_query")
                    .set_vuids(&["VUID-VkCommandBufferInheritanceInfo-queryFlags-00057"])
            })?;

            if !device.enabled_features().inherited_queries {
                return Err(Box::new(ValidationError {
                    context: "occlusion_query".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "inherited_queries",
                    )])]),
                    vuids: &["VUID-VkCommandBufferInheritanceInfo-occlusionQueryEnable-00056"],
                }));
            }

            if control_flags.intersects(QueryControlFlags::PRECISE)
                && !device.enabled_features().occlusion_query_precise
            {
                return Err(Box::new(ValidationError {
                    context: "occlusion_query".into(),
                    problem: "contains `QueryControlFlags::PRECISE`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "occlusion_query_precise",
                    )])]),
                    vuids: &["VUID-vkBeginCommandBuffer-commandBuffer-00052"],
                }));
            }
        }

        pipeline_statistics.validate_device(device).map_err(|err| {
            err.add_context("pipeline_statistics")
                .set_vuids(&["VUID-VkCommandBufferInheritanceInfo-pipelineStatistics-02789"])
        })?;

        if pipeline_statistics.count() > 0 && !device.enabled_features().pipeline_statistics_query {
            return Err(Box::new(ValidationError {
                context: "pipeline_statistics".into(),
                problem: "is not empty".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "pipeline_statistics_query",
                )])]),
                vuids: &["VUID-VkCommandBufferInheritanceInfo-pipelineStatistics-00058"],
            }));
        }

        Ok(())
    }
}

/// Selects the type of render pass for command buffer inheritance.
#[derive(Clone, Debug)]
pub enum CommandBufferInheritanceRenderPassType {
    /// The secondary command buffer will be executed within a render pass begun with
    /// `begin_render_pass`, using a `RenderPass` object and `Framebuffer`.
    BeginRenderPass(CommandBufferInheritanceRenderPassInfo),

    /// The secondary command buffer will be executed within a render pass begun with
    /// `begin_rendering`, using dynamic rendering.
    BeginRendering(CommandBufferInheritanceRenderingInfo),
}

impl From<Subpass> for CommandBufferInheritanceRenderPassType {
    #[inline]
    fn from(val: Subpass) -> Self {
        Self::BeginRenderPass(val.into())
    }
}

impl From<CommandBufferInheritanceRenderPassInfo> for CommandBufferInheritanceRenderPassType {
    #[inline]
    fn from(val: CommandBufferInheritanceRenderPassInfo) -> Self {
        Self::BeginRenderPass(val)
    }
}

impl From<CommandBufferInheritanceRenderingInfo> for CommandBufferInheritanceRenderPassType {
    #[inline]
    fn from(val: CommandBufferInheritanceRenderingInfo) -> Self {
        Self::BeginRendering(val)
    }
}

/// The render pass context that a secondary command buffer is created for.
#[derive(Clone, Debug)]
pub struct CommandBufferInheritanceRenderPassInfo {
    /// The render subpass that this secondary command buffer must be executed within.
    ///
    /// There is no default value.
    pub subpass: Subpass,

    /// The framebuffer object that will be used when calling the command buffer.
    /// This parameter is optional and is an optimization hint for the implementation.
    ///
    /// The default value is `None`.
    pub framebuffer: Option<Arc<Framebuffer>>,
}

impl CommandBufferInheritanceRenderPassInfo {
    /// Returns a `CommandBufferInheritanceRenderPassInfo` with the specified `subpass`.
    #[inline]
    pub fn subpass(subpass: Subpass) -> Self {
        Self {
            subpass,
            framebuffer: None,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref subpass,
            ref framebuffer,
        } = self;

        // VUID-VkCommandBufferInheritanceInfo-commonparent
        assert_eq!(device, subpass.render_pass().device().as_ref());

        // VUID-VkCommandBufferBeginInfo-flags-06001
        // Ensured by how the `Subpass` type is constructed.

        if let Some(framebuffer) = framebuffer {
            // VUID-VkCommandBufferInheritanceInfo-commonparent
            assert_eq!(device, framebuffer.device().as_ref());

            if !framebuffer
                .render_pass()
                .is_compatible_with(subpass.render_pass())
            {
                return Err(Box::new(ValidationError {
                    problem: "`framebuffer` is not compatible with `subpass.render_pass()`".into(),
                    vuids: &["VUID-VkCommandBufferBeginInfo-flags-00055"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }
}

impl From<Subpass> for CommandBufferInheritanceRenderPassInfo {
    #[inline]
    fn from(subpass: Subpass) -> Self {
        Self {
            subpass,
            framebuffer: None,
        }
    }
}

/// The dynamic rendering context that a secondary command buffer is created for.
#[derive(Clone, Debug)]
pub struct CommandBufferInheritanceRenderingInfo {
    /// If not `0`, indicates that multiview rendering will be enabled, and specifies the view
    /// indices that are rendered to. The value is a bitmask, so that that for example `0b11` will
    /// draw to the first two views and `0b101` will draw to the first and third view.
    ///
    /// If set to a nonzero value, then the [`multiview`] feature must be enabled on the device.
    ///
    /// The default value is `0`.
    ///
    /// [`multiview`]: crate::device::DeviceFeatures::multiview
    pub view_mask: u32,

    /// The formats of the color attachments that will be used during rendering.
    ///
    /// If an element is `None`, it indicates that the attachment will not be used.
    ///
    /// The default value is empty.
    pub color_attachment_formats: Vec<Option<Format>>,

    /// The format of the depth attachment that will be used during rendering.
    ///
    /// If set to `None`, it indicates that no depth attachment will be used.
    ///
    /// The default value is `None`.
    pub depth_attachment_format: Option<Format>,

    /// The format of the stencil attachment that will be used during rendering.
    ///
    /// If set to `None`, it indicates that no stencil attachment will be used.
    ///
    /// The default value is `None`.
    pub stencil_attachment_format: Option<Format>,

    /// The number of samples that the color, depth and stencil attachments will have.
    ///
    /// The default value is [`SampleCount::Sample1`]
    pub rasterization_samples: SampleCount,
}

impl Default for CommandBufferInheritanceRenderingInfo {
    #[inline]
    fn default() -> Self {
        Self {
            view_mask: 0,
            color_attachment_formats: Vec::new(),
            depth_attachment_format: None,
            stencil_attachment_format: None,
            rasterization_samples: SampleCount::Sample1,
        }
    }
}

impl CommandBufferInheritanceRenderingInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            view_mask,
            ref color_attachment_formats,
            depth_attachment_format,
            stencil_attachment_format,
            rasterization_samples,
        } = self;

        let properties = device.physical_device().properties();

        if view_mask != 0 && !device.enabled_features().multiview {
            return Err(Box::new(ValidationError {
                context: "view_mask".into(),
                problem: "is not zero".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "multiview",
                )])]),
                vuids: &["VUID-VkCommandBufferInheritanceRenderingInfo-multiview-06008"],
            }));
        }

        let view_count = u32::BITS - view_mask.leading_zeros();

        if view_count > properties.max_multiview_view_count.unwrap_or(0) {
            return Err(Box::new(ValidationError {
                context: "view_mask".into(),
                problem: "the number of views exceeds the \
                    `max_multiview_view_count` limit"
                    .into(),
                vuids: &["VUID-VkCommandBufferInheritanceRenderingInfo-viewMask-06009"],
                ..Default::default()
            }));
        }

        for (index, format) in color_attachment_formats
            .iter()
            .enumerate()
            .flat_map(|(i, f)| f.map(|f| (i, f)))
        {
            format.validate_device(device).map_err(|err| {
                err.add_context(format!("color_attachment_formats[{}]", index)).set_vuids(
                    &["VUID-VkCommandBufferInheritanceRenderingInfo-pColorAttachmentFormats-parameter"],
                )
            })?;

            if format == Format::UNDEFINED {
                return Err(Box::new(ValidationError {
                    context: format!("color_attachment_formats[{}]", index).into(),
                    problem: "is `Format::UNDEFINED`".into(),
                    ..Default::default()
                }));
            }

            let potential_format_features = unsafe {
                device
                    .physical_device()
                    .format_properties_unchecked(format)
                    .potential_format_features()
            };

            if !potential_format_features.intersects(FormatFeatures::COLOR_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: format!("color_attachment_formats[{}]", index).into(),
                    problem: "the potential format features do not contain \
                        `FormatFeatures::COLOR_ATTACHMENT`".into(),
                    vuids: &["VUID-VkCommandBufferInheritanceRenderingInfo-pColorAttachmentFormats-06006"],
                    ..Default::default()
                }));
            }
        }

        if let Some(format) = depth_attachment_format {
            format.validate_device(device).map_err(|err| {
                err.add_context("depth_attachment_format").set_vuids(&[
                    "VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-parameter",
                ])
            })?;

            if format == Format::UNDEFINED {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment_format".into(),
                    problem: "is `Format::UNDEFINED`".into(),
                    ..Default::default()
                }));
            }

            if !format.aspects().intersects(ImageAspects::DEPTH) {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment_format".into(),
                    problem: "does not have a depth aspect".into(),
                    vuids: &[
                        "VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06540",
                    ],
                    ..Default::default()
                }));
            }

            let potential_format_features = unsafe {
                device
                    .physical_device()
                    .format_properties_unchecked(format)
                    .potential_format_features()
            };

            if !potential_format_features.intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment_format".into(),
                    problem: "the potential format features do not contain \
                        `FormatFeatures::DEPTH_STENCIL_ATTACHMENT`"
                        .into(),
                    vuids: &[
                        "VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06007",
                    ],
                    ..Default::default()
                }));
            }
        }

        if let Some(format) = stencil_attachment_format {
            format.validate_device(device).map_err(|err| {
                err.add_context("stencil_attachment_format").set_vuids(&["VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-parameter"])
            })?;

            if format == Format::UNDEFINED {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment_format".into(),
                    problem: "is `Format::UNDEFINED`".into(),
                    ..Default::default()
                }));
            }

            if !format.aspects().intersects(ImageAspects::STENCIL) {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment_format".into(),
                    problem: "does not have a stencil aspect".into(),
                    vuids: &[
                        "VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-06541",
                    ],
                    ..Default::default()
                }));
            }

            let potential_format_features = unsafe {
                device
                    .physical_device()
                    .format_properties_unchecked(format)
                    .potential_format_features()
            };

            if !potential_format_features.intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment_format".into(),
                    problem: "the potential format features do not contain \
                        `FormatFeatures::DEPTH_STENCIL_ATTACHMENT`"
                        .into(),
                    vuids: &[
                        "VUID-VkCommandBufferInheritanceRenderingInfo-stencilAttachmentFormat-06199",
                    ],
                    ..Default::default()
                }));
            }
        }

        if let (Some(depth_format), Some(stencil_format)) =
            (depth_attachment_format, stencil_attachment_format)
        {
            if depth_format != stencil_format {
                return Err(Box::new(ValidationError {
                    problem: "`depth_attachment_format` and `stencil_attachment_format` are both \
                        `Some`, but are not equal"
                        .into(),
                    vuids: &[
                        "VUID-VkCommandBufferInheritanceRenderingInfo-depthAttachmentFormat-06200",
                    ],
                    ..Default::default()
                }));
            }
        }

        rasterization_samples
            .validate_device(device)
            .map_err(|err| {
                err.add_context("rasterization_samples").set_vuids(&[
                    "VUID-VkCommandBufferInheritanceRenderingInfo-rasterizationSamples-parameter",
                ])
            })?;

        Ok(())
    }
}

/// Usage flags to pass when creating a command buffer.
///
/// The safest option is `SimultaneousUse`, but it may be slower than the other two.
// NOTE: The ordering is important: the variants are listed from least to most permissive!
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum CommandBufferUsage {
    /// The command buffer can only be submitted once before being destroyed. Any further submit is
    /// forbidden. This makes it possible for the implementation to perform additional
    /// optimizations.
    OneTimeSubmit = ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT.as_raw(),

    /// The command buffer can be used multiple times, but must not execute or record more than
    /// once simultaneously. In other words, it is as if executing the command buffer borrows
    /// it mutably.
    MultipleSubmit = 0,

    /// The command buffer can be executed multiple times in parallel on different queues.
    /// If it's a secondary command buffer, it can be recorded to multiple primary command buffers
    /// at once.
    SimultaneousUse = ash::vk::CommandBufferUsageFlags::SIMULTANEOUS_USE.as_raw(),
}

impl From<CommandBufferUsage> for ash::vk::CommandBufferUsageFlags {
    #[inline]
    fn from(val: CommandBufferUsage) -> Self {
        Self::from_raw(val as u32)
    }
}

/// Parameters to submit command buffers to a queue.
#[derive(Clone, Debug)]
pub struct SubmitInfo {
    /// The semaphores to wait for before beginning the execution of this batch of
    /// command buffer operations.
    ///
    /// The default value is empty.
    pub wait_semaphores: Vec<SemaphoreSubmitInfo>,

    /// The command buffers to execute.
    ///
    /// The default value is empty.
    pub command_buffers: Vec<CommandBufferSubmitInfo>,

    /// The semaphores to signal after the execution of this batch of command buffer operations
    /// has completed.
    ///
    /// The default value is empty.
    pub signal_semaphores: Vec<SemaphoreSubmitInfo>,

    pub _ne: crate::NonExhaustive,
}

impl Default for SubmitInfo {
    #[inline]
    fn default() -> Self {
        Self {
            wait_semaphores: Vec::new(),
            command_buffers: Vec::new(),
            signal_semaphores: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SubmitInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref wait_semaphores,
            ref command_buffers,
            ref signal_semaphores,
            _ne: _,
        } = self;

        for (index, semaphore_submit_info) in wait_semaphores.iter().enumerate() {
            semaphore_submit_info
                .validate(device)
                .map_err(|err| err.add_context(format!("wait_semaphores[{}]", index)))?;
        }

        for (index, command_buffer_submit_info) in command_buffers.iter().enumerate() {
            command_buffer_submit_info
                .validate(device)
                .map_err(|err| err.add_context(format!("command_buffers[{}]", index)))?;
        }

        for (index, semaphore_submit_info) in signal_semaphores.iter().enumerate() {
            semaphore_submit_info
                .validate(device)
                .map_err(|err| err.add_context(format!("signal_semaphores[{}]", index)))?;

            let &SemaphoreSubmitInfo {
                semaphore: _,
                value: _,
                stages,
                _ne: _,
            } = semaphore_submit_info;

            if stages != PipelineStages::ALL_COMMANDS && !device.enabled_features().synchronization2
            {
                return Err(Box::new(ValidationError {
                    context: format!("signal_semaphores[{}].stages", index).into(),
                    problem: "is not `PipelineStages::ALL_COMMANDS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    vuids: &["VUID-vkQueueSubmit2-synchronization2-03866"],
                }));
            }
        }

        // unsafe
        // VUID-VkSubmitInfo2-semaphore-03882
        // VUID-VkSubmitInfo2-semaphore-03883
        // VUID-VkSubmitInfo2-semaphore-03884

        Ok(())
    }
}

/// Parameters for a command buffer in a queue submit operation.
#[derive(Clone, Debug)]
pub struct CommandBufferSubmitInfo {
    /// The command buffer to execute.
    ///
    /// There is no default value.
    pub command_buffer: Arc<CommandBuffer>,

    pub _ne: crate::NonExhaustive,
}

impl CommandBufferSubmitInfo {
    /// Returns a `CommandBufferSubmitInfo` with the specified `command_buffer`.
    #[inline]
    pub fn new(command_buffer: Arc<CommandBuffer>) -> Self {
        Self {
            command_buffer,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref command_buffer,
            _ne: _,
        } = self;

        // VUID?
        assert_eq!(device, command_buffer.device().as_ref());

        if self.command_buffer.level() != CommandBufferLevel::Primary {
            return Err(Box::new(ValidationError {
                context: "command_buffer".into(),
                problem: "is not a primary command buffer".into(),
                vuids: &["VUID-VkCommandBufferSubmitInfo-commandBuffer-03890"],
                ..Default::default()
            }));
        }

        Ok(())
    }
}

/// Parameters for a semaphore signal or wait operation in a queue submit operation.
#[derive(Clone, Debug)]
pub struct SemaphoreSubmitInfo {
    /// The semaphore to signal or wait for.
    ///
    /// There is no default value.
    pub semaphore: Arc<Semaphore>,

    /// If `semaphore.semaphore_type()` is [`SemaphoreType::Timeline`], specifies the value that
    /// will be used for the semaphore operation:
    /// - If it's a signal operation, then the semaphore's value will be set to this value when it
    ///   is signaled.
    /// - If it's a wait operation, then the semaphore will wait until its value is greater than or
    ///   equal to this value.
    ///
    /// If `semaphore.semaphore_type()` is [`SemaphoreType::Binary`], then this must be `0`.
    ///
    /// The default value is `0`.
    pub value: u64,

    /// For a semaphore wait operation, specifies the pipeline stages in the second synchronization
    /// scope: stages of queue operations following the wait operation that can start executing
    /// after the semaphore is signalled.
    ///
    /// For a semaphore signal operation, specifies the pipeline stages in the first
    /// synchronization scope: stages of queue operations preceding the signal operation that
    /// must complete before the semaphore is signalled.
    /// If this value does not equal [`ALL_COMMANDS`], then the [`synchronization2`] feature must
    /// be enabled on the device.
    ///
    /// The default value is [`ALL_COMMANDS`].
    ///
    /// [`ALL_COMMANDS`]: PipelineStages::ALL_COMMANDS
    /// [`synchronization2`]: crate::device::DeviceFeatures::synchronization2
    pub stages: PipelineStages,

    pub _ne: crate::NonExhaustive,
}

impl SemaphoreSubmitInfo {
    /// Returns a `SemaphoreSubmitInfo` with the specified `semaphore`.
    #[inline]
    pub fn new(semaphore: Arc<Semaphore>) -> Self {
        Self {
            semaphore,
            value: 0,
            stages: PipelineStages::ALL_COMMANDS,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref semaphore,
            value,
            stages,
            _ne: _,
        } = self;

        // VUID?
        assert_eq!(device, semaphore.device().as_ref());

        match semaphore.semaphore_type() {
            SemaphoreType::Binary => {
                if value != 0 {
                    return Err(Box::new(ValidationError {
                        problem: "`semaphore.semaphore_type()` is `SemaphoreType::Binary`, but \
                            `value` is not `0`"
                            .into(),
                        ..Default::default()
                    }));
                }
            }
            SemaphoreType::Timeline => {}
        }

        stages.validate_device(device).map_err(|err| {
            err.add_context("stages")
                .set_vuids(&["VUID-VkSemaphoreSubmitInfo-stageMask-parameter"])
        })?;

        if !device.enabled_features().synchronization2 && stages.contains_flags2() {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "synchronization2",
                )])]),
                ..Default::default()
            }));
        }

        if !device.enabled_features().geometry_shader
            && stages.intersects(PipelineStages::GEOMETRY_SHADER)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "geometry_shader",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03929"],
            }));
        }

        if !device.enabled_features().tessellation_shader
            && stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            )
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                    `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "tessellation_shader",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03930"],
            }));
        }

        if !device.enabled_features().conditional_rendering
            && stages.intersects(PipelineStages::CONDITIONAL_RENDERING)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "conditional_rendering",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03931"],
            }));
        }

        if !device.enabled_features().fragment_density_map
            && stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "fragment_density_map",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03932"],
            }));
        }

        if !device.enabled_features().transform_feedback
            && stages.intersects(PipelineStages::TRANSFORM_FEEDBACK)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "transform_feedback",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03933"],
            }));
        }

        if !device.enabled_features().mesh_shader && stages.intersects(PipelineStages::MESH_SHADER)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::MESH_SHADER`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "mesh_shader",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03934"],
            }));
        }

        if !device.enabled_features().task_shader && stages.intersects(PipelineStages::TASK_SHADER)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::TASK_SHADER`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "task_shader",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-03935"],
            }));
        }

        if !(device.enabled_features().attachment_fragment_shading_rate
            || device.enabled_features().shading_rate_image)
            && stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::DeviceFeature("attachment_fragment_shading_rate")]),
                    RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                ]),
                vuids: &["VUID-VkMemoryBarrier2-shadingRateImage-07316"],
            }));
        }

        if !device.enabled_features().subpass_shading
            && stages.intersects(PipelineStages::SUBPASS_SHADING)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::SUBPASS_SHADING`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "subpass_shading",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-04957"],
            }));
        }

        if !device.enabled_features().invocation_mask
            && stages.intersects(PipelineStages::INVOCATION_MASK)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "invocation_mask",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-04995"],
            }));
        }

        if !(device.enabled_extensions().nv_ray_tracing
            || device.enabled_features().ray_tracing_pipeline)
            && stages.intersects(PipelineStages::RAY_TRACING_SHADER)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "ray_tracing_pipeline",
                )])]),
                vuids: &["VUID-VkSemaphoreSubmitInfo-stageMask-07946"],
            }));
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct CommandBufferState {
    has_been_submitted: bool,
    pending_submits: u32,
}

impl CommandBufferState {
    pub(crate) fn has_been_submitted(&self) -> bool {
        self.has_been_submitted
    }

    pub(crate) fn is_submit_pending(&self) -> bool {
        self.pending_submits != 0
    }

    pub(crate) unsafe fn add_queue_submit(&mut self) {
        self.has_been_submitted = true;
        self.pending_submits += 1;
    }

    pub(crate) unsafe fn set_submit_finished(&mut self) {
        self.pending_submits -= 1;
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct CommandBufferResourcesUsage {
    pub(crate) buffers: Vec<CommandBufferBufferUsage>,
    pub(crate) images: Vec<CommandBufferImageUsage>,
    pub(crate) buffer_indices: HashMap<Arc<Buffer>, usize>,
    pub(crate) image_indices: HashMap<Arc<Image>, usize>,
}

#[derive(Debug)]
pub(crate) struct CommandBufferBufferUsage {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) ranges: RangeMap<DeviceSize, CommandBufferBufferRangeUsage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CommandBufferBufferRangeUsage {
    pub(crate) first_use: Option<ResourceUseRef>,
    pub(crate) mutable: bool,
}

#[derive(Debug)]
pub(crate) struct CommandBufferImageUsage {
    pub(crate) image: Arc<Image>,
    pub(crate) ranges: RangeMap<DeviceSize, CommandBufferImageRangeUsage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CommandBufferImageRangeUsage {
    pub(crate) first_use: Option<ResourceUseRef>,
    pub(crate) mutable: bool,
    pub(crate) expected_layout: ImageLayout,
    pub(crate) final_layout: ImageLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceUseRef {
    pub command_index: usize,
    pub command_name: &'static str,
    pub resource_in_command: ResourceInCommand,
    pub secondary_use_ref: Option<SecondaryResourceUseRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SecondaryResourceUseRef {
    pub command_index: usize,
    pub command_name: &'static str,
    pub resource_in_command: ResourceInCommand,
}

impl From<ResourceUseRef> for SecondaryResourceUseRef {
    #[inline]
    fn from(val: ResourceUseRef) -> Self {
        let ResourceUseRef {
            command_index,
            command_name,
            resource_in_command,
            secondary_use_ref,
        } = val;

        debug_assert!(secondary_use_ref.is_none());

        SecondaryResourceUseRef {
            command_index,
            command_name,
            resource_in_command,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResourceInCommand {
    AccelerationStructure { index: u32 },
    ColorAttachment { index: u32 },
    ColorResolveAttachment { index: u32 },
    DepthStencilAttachment,
    DepthStencilResolveAttachment,
    DescriptorSet { set: u32, binding: u32, index: u32 },
    Destination,
    FramebufferAttachment { index: u32 },
    GeometryAabbsData { index: u32 },
    GeometryInstancesData,
    GeometryTrianglesTransformData { index: u32 },
    GeometryTrianglesIndexData { index: u32 },
    GeometryTrianglesVertexData { index: u32 },
    ImageMemoryBarrier { index: u32 },
    IndexBuffer,
    IndirectBuffer,
    ScratchData,
    SecondaryCommandBuffer { index: u32 },
    Source,
    VertexBuffer { binding: u32 },
}

#[doc(hidden)]
#[derive(Debug, Default)]
pub struct SecondaryCommandBufferResourcesUsage {
    pub(crate) buffers: Vec<SecondaryCommandBufferBufferUsage>,
    pub(crate) images: Vec<SecondaryCommandBufferImageUsage>,
}

#[derive(Debug)]
pub(crate) struct SecondaryCommandBufferBufferUsage {
    pub(crate) use_ref: ResourceUseRef,
    pub(crate) buffer: Subbuffer<[u8]>,
    pub(crate) range: Range<DeviceSize>,
    pub(crate) memory_access: PipelineStageAccessFlags,
}

#[derive(Debug)]
pub(crate) struct SecondaryCommandBufferImageUsage {
    pub(crate) use_ref: ResourceUseRef,
    pub(crate) image: Arc<Image>,
    pub(crate) subresource_range: ImageSubresourceRange,
    pub(crate) memory_access: PipelineStageAccessFlags,
    pub(crate) start_layout: ImageLayout,
    pub(crate) end_layout: ImageLayout,
}
