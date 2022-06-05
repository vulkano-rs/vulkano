// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Commands that the GPU will execute (includes draw commands).
//!
//! With Vulkan, before the GPU can do anything you must create a `CommandBuffer`. A command buffer
//! is a list of commands that will executed by the GPU. Once a command buffer is created, you can
//! execute it. A command buffer must always be created even for the most simple tasks.
//!
//! # Primary and secondary command buffers.
//!
//! There are three types of command buffers:
//!
//! - **Primary command buffers**. They can contain any command. They are the only type of command
//!   buffer that can be submitted to a queue.
//! - **Secondary "graphics" command buffers**. They can only contain draw and clear commands.
//!   They can only be called from a primary command buffer when inside a render pass.
//! - **Secondary "compute" command buffers**. They can only contain non-render-pass-related
//!   commands (ie. everything but drawing, clearing, etc.) and cannot enter a render pass. They
//!   can only be called from a primary command buffer outside of a render pass.
//!
//! Using secondary command buffers leads to slightly lower performance on the GPU, but they have
//! two advantages on the CPU side:
//!
//! - Building a command buffer is a single-threaded operation, but by using secondary command
//!   buffers you can build multiple secondary command buffers in multiple threads simultaneously.
//! - Secondary command buffers can be kept alive between frames. When you always repeat the same
//!   operations, it might be a good idea to build a secondary command buffer once at
//!   initialization and then reuse it afterwards.
//!
//! # The `AutoCommandBufferBuilder`
//!
//! The most basic (and recommended) way to create a command buffer is to create a
//! [`AutoCommandBufferBuilder`], then record commands to it.
//! When you are done adding commands, build it to obtain either a `PrimaryAutoCommandBuffer` or
//! `SecondAutoCommandBuffer`.
//!
//! Once built, use [the `PrimaryCommandBuffer` trait](crate::command_buffer::PrimaryCommandBuffer) to submit the
//! command buffer. Submitting a command buffer returns an object that implements the `GpuFuture` trait
//! and that represents the moment when the execution will end on the GPU.
//!
//! ```
//! use vulkano::command_buffer::AutoCommandBufferBuilder;
//! use vulkano::command_buffer::CommandBufferUsage;
//! use vulkano::command_buffer::PrimaryCommandBuffer;
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! # let queue: std::sync::Arc<vulkano::device::Queue> = return;
//! let cb = AutoCommandBufferBuilder::primary(
//!     device.clone(),
//!     queue.family(),
//!     CommandBufferUsage::MultipleSubmit
//! ).unwrap()
//! // TODO: add an actual command to this example
//! .build().unwrap();
//!
//! let _future = cb.execute(queue.clone());
//! ```
//!
//! # Internal architecture of vulkano
//!
//! The `commands_raw` and `commands_extra` modules contain structs that correspond to various
//! commands that can be added to command buffer builders. A command can be added to a command
//! buffer builder by using the `AddCommand<C>` trait, where `C` is the command struct.
//!
//! The `AutoCommandBufferBuilder` internally uses a `UnsafeCommandBufferBuilder` wrapped around
//! multiple layers. See the `cb` module for more information.
//!
//! Command pools are automatically handled by default, but vulkano also allows you to use
//! alternative command pool implementations and use them. See the `pool` module for more
//! information.

pub use self::commands::{
    debug::DebugUtilsError,
    image::{
        BlitImageInfo, ClearColorImageInfo, ClearDepthStencilImageInfo, ImageBlit, ImageResolve,
        ResolveImageInfo,
    },
    pipeline::{
        CheckDescriptorSetsValidityError, CheckDispatchError, CheckDynamicStateValidityError,
        CheckIndexBufferError, CheckIndirectBufferError, CheckPipelineError,
        CheckPushConstantsValidityError, CheckVertexBufferError,
    },
    query::QueryError,
    render_pass::{
        ClearAttachment, ClearRect, RenderPassBeginInfo, RenderPassError, RenderingAttachmentInfo,
        RenderingAttachmentResolveInfo, RenderingInfo,
    },
    secondary::{ExecuteCommandsError, UnsafeCommandBufferBuilderExecuteCommands},
    transfer::{
        BufferCopy, BufferImageCopy, CopyBufferInfo, CopyBufferInfoTyped, CopyBufferToImageInfo,
        CopyImageInfo, CopyImageToBufferInfo, FillBufferInfo, ImageCopy,
    },
    CopyError, CopyErrorResource,
};
pub use self::{
    auto::{
        AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, BuildError,
        CommandBufferBeginError, DispatchError, DispatchIndirectError, DrawError, DrawIndexedError,
        DrawIndexedIndirectError, DrawIndirectError, PrimaryAutoCommandBuffer,
        SecondaryAutoCommandBuffer,
    },
    traits::{
        CommandBufferExecError, CommandBufferExecFuture, PrimaryCommandBuffer,
        SecondaryCommandBuffer,
    },
};
use crate::{
    format::Format,
    image::SampleCount,
    query::{QueryControlFlags, QueryPipelineStatisticFlags},
    render_pass::{Framebuffer, Subpass},
};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

mod auto;
mod commands;
pub mod pool;
pub mod submit;
pub mod synced;
pub mod sys;
mod traits;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, PartialEq, Eq)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, PartialEq, Eq)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: u32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, PartialEq, Eq)]
pub struct DispatchIndirectCommand {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Describes what a subpass in a command buffer will contain.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum SubpassContents {
    /// The subpass will only directly contain commands.
    Inline = ash::vk::SubpassContents::INLINE.as_raw(),
    /// The subpass will only contain secondary command buffers invocations.
    SecondaryCommandBuffers = ash::vk::SubpassContents::SECONDARY_COMMAND_BUFFERS.as_raw(),
}

impl From<SubpassContents> for ash::vk::SubpassContents {
    #[inline]
    fn from(val: SubpassContents) -> Self {
        Self::from_raw(val as i32)
    }
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

/// Determines the kind of command buffer to create.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum CommandBufferLevel {
    /// Primary command buffers can be executed on a queue, and can call secondary command buffers.
    /// Render passes must begin and end within the same primary command buffer.
    Primary = ash::vk::CommandBufferLevel::PRIMARY.as_raw(),

    /// Secondary command buffers cannot be executed on a queue, but can be executed by a primary
    /// command buffer. If created for a render pass, they must fit within a single render subpass.
    Secondary = ash::vk::CommandBufferLevel::SECONDARY.as_raw(),
}

impl From<CommandBufferLevel> for ash::vk::CommandBufferLevel {
    #[inline]
    fn from(val: CommandBufferLevel) -> Self {
        Self::from_raw(val as i32)
    }
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

    /// Which pipeline statistics queries are allowed to be active on the primary command buffer
    /// when this secondary command buffer is executed.
    ///
    /// The `pipeline_statistics_query` feature must be enabled if any of the flags of this value
    /// are set.
    ///
    /// The default value is [`QueryPipelineStatisticFlags::none()`].
    pub query_statistics_flags: QueryPipelineStatisticFlags,

    pub _ne: crate::NonExhaustive,
}

impl Default for CommandBufferInheritanceInfo {
    #[inline]
    fn default() -> Self {
        Self {
            render_pass: None,
            occlusion_query: None,
            query_statistics_flags: QueryPipelineStatisticFlags::none(),
            _ne: crate::NonExhaustive(()),
        }
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
    /// If set to a nonzero value, the [`multiview`](crate::device::Features::multiview) feature
    /// must be enabled on the device.
    ///
    /// The default value is `0`.
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

    /// The command buffer can be used multiple times, but must not execute or record more than once
    /// simultaneously. In other words, it is as if executing the command buffer borrows it mutably.
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
