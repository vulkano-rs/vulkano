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
//! [`AutoCommandBufferBuilder`](struct.AutoCommandBufferBuilder.html), then record commands to it.
//! When you are done adding commands, build it to obtain either a `PrimaryAutoCommandBuffer` or
//! `SecondAutoCommandBuffer`.
//!
//! Once built, use [the `PrimaryCommandBuffer` trait](trait.PrimaryCommandBuffer.html) to submit the
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

pub use self::auto::AutoCommandBufferBuilder;
pub use self::auto::AutoCommandBufferBuilderContextError;
pub use self::auto::BeginError;
pub use self::auto::BeginQueryError;
pub use self::auto::BeginRenderPassError;
pub use self::auto::BlitImageError;
pub use self::auto::BuildError;
pub use self::auto::ClearColorImageError;
pub use self::auto::CopyBufferError;
pub use self::auto::CopyBufferImageError;
pub use self::auto::CopyImageError;
pub use self::auto::CopyQueryPoolResultsError;
pub use self::auto::DebugMarkerError;
pub use self::auto::DispatchError;
pub use self::auto::DispatchIndirectError;
pub use self::auto::DrawError;
pub use self::auto::DrawIndexedError;
pub use self::auto::DrawIndexedIndirectError;
pub use self::auto::DrawIndirectError;
pub use self::auto::EndQueryError;
pub use self::auto::ExecuteCommandsError;
pub use self::auto::FillBufferError;
pub use self::auto::PrimaryAutoCommandBuffer;
pub use self::auto::ResetQueryPoolError;
pub use self::auto::SecondaryAutoCommandBuffer;
pub use self::auto::UpdateBufferError;
pub use self::auto::WriteTimestampError;
pub use self::state_cacher::StateCacher;
pub use self::state_cacher::StateCacherOutcome;
pub use self::traits::CommandBufferExecError;
pub use self::traits::CommandBufferExecFuture;
pub use self::traits::PrimaryCommandBuffer;
pub use self::traits::SecondaryCommandBuffer;
use crate::pipeline::depth_stencil::DynamicStencilValue;
use crate::pipeline::viewport::{Scissor, Viewport};
use crate::query::QueryControlFlags;
use crate::query::QueryPipelineStatisticFlags;
use crate::render_pass::{Framebuffer, Subpass};
use std::sync::Arc;

mod auto;
pub mod pool;
mod state_cacher;
pub mod submit;
pub mod synced;
pub mod sys;
mod traits;
pub mod validity;

#[derive(Debug, Clone, Copy)]
pub enum ImageUninitializedSafe {
    Safe,
    Unsafe,
}

impl ImageUninitializedSafe {
    pub fn is_safe(&self) -> bool {
        match self {
            Self::Safe => true,
            Self::Unsafe => false,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: u32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DispatchIndirectCommand {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// The dynamic state to use for a draw command.
// TODO: probably not the right location
#[derive(Debug, Clone)]
pub struct DynamicState {
    pub line_width: Option<f32>,
    pub viewports: Option<Vec<Viewport>>,
    pub scissors: Option<Vec<Scissor>>,
    pub compare_mask: Option<DynamicStencilValue>,
    pub write_mask: Option<DynamicStencilValue>,
    pub reference: Option<DynamicStencilValue>,
}

impl DynamicState {
    #[inline]
    pub fn none() -> DynamicState {
        DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
            compare_mask: None,
            write_mask: None,
            reference: None,
        }
    }
}

impl Default for DynamicState {
    #[inline]
    fn default() -> DynamicState {
        DynamicState::none()
    }
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

/// Determines the kind of command buffer to create.
#[derive(Debug, Clone)]
pub enum CommandBufferLevel<F> {
    /// Primary command buffers can be executed on a queue, and can call secondary command buffers.
    /// Render passes must begin and end within the same primary command buffer.
    Primary,

    /// Secondary command buffers cannot be executed on a queue, but can be executed by a primary
    /// command buffer. If created for a render pass, they must fit within a single render subpass.
    Secondary(CommandBufferInheritance<F>),
}

/// The context that a secondary command buffer can inherit from the primary command
/// buffer it's executed in.
#[derive(Clone, Debug, Default)]
pub struct CommandBufferInheritance<F> {
    /// If `Some`, the secondary command buffer is required to be executed within a specific
    /// render subpass, and can only call draw operations.
    /// If `None`, it must be executed outside a render pass, and can execute dispatch and transfer
    /// operations, but not drawing operations.
    render_pass: Option<CommandBufferInheritanceRenderPass<F>>,

    /// If `Some`, the secondary command buffer is allowed to be executed within a primary that has
    /// an occlusion query active. The inner `QueryControlFlags` specifies which flags the
    /// active occlusion is allowed to have enabled.
    /// If `None`, the primary command buffer cannot have an occlusion query active when this
    /// secondary command buffer is executed.
    ///
    /// The `inherited_queries` feature must be enabled if this is `Some`.
    occlusion_query: Option<QueryControlFlags>,

    /// Which pipeline statistics queries are allowed to be active on the primary command buffer
    /// when this secondary command buffer is executed.
    ///
    /// The `pipeline_statistics_query` feature must be enabled if any of the flags of this value
    /// are set.
    query_statistics_flags: QueryPipelineStatisticFlags,
}

/// The render pass context that a secondary command buffer is created for.
#[derive(Debug, Clone)]
pub struct CommandBufferInheritanceRenderPass<F> {
    /// The render subpass that this secondary command buffer must be executed within.
    pub subpass: Subpass,

    /// The framebuffer object that will be used when calling the command buffer.
    /// This parameter is optional and is an optimization hint for the implementation.
    pub framebuffer: Option<F>,
}

impl CommandBufferLevel<Framebuffer<()>> {
    /// Equivalent to `Kind::Primary`.
    ///
    /// > **Note**: If you use `let kind = Kind::Primary;` in your code, you will probably get a
    /// > compilation error because the Rust compiler couldn't determine the template parameters
    /// > of `Kind`. To solve that problem in an easy way you can use this function instead.
    #[inline]
    pub fn primary() -> CommandBufferLevel<Arc<Framebuffer<()>>> {
        CommandBufferLevel::Primary
    }

    /// Equivalent to `Kind::Secondary`.
    ///
    /// > **Note**: If you use `let kind = Kind::Secondary;` in your code, you will probably get a
    /// > compilation error because the Rust compiler couldn't determine the template parameters
    /// > of `Kind`. To solve that problem in an easy way you can use this function instead.
    #[inline]
    pub fn secondary(
        occlusion_query: Option<QueryControlFlags>,
        query_statistics_flags: QueryPipelineStatisticFlags,
    ) -> CommandBufferLevel<Arc<Framebuffer<()>>> {
        CommandBufferLevel::Secondary(CommandBufferInheritance {
            render_pass: None,
            occlusion_query,
            query_statistics_flags,
        })
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
