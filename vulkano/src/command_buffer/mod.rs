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
//! [`AutoCommandBufferBuilder`](struct.AutoCommandBufferBuilder.html). Then use the
//! [`CommandBufferBuilder` trait](trait.CommandBufferBuilder.html) to add commands to it.
//! When you are done adding commands, use
//! [the `CommandBufferBuild` trait](trait.CommandBufferBuild.html) to obtain a
//! `AutoCommandBuffer`.
//!
//! Once built, use [the `CommandBuffer` trait](trait.CommandBuffer.html) to submit the command
//! buffer. Submitting a command buffer returns an object that implements the `GpuFuture` trait and
//! that represents the moment when the execution will end on the GPU.
//!
//! ```
//! use vulkano::command_buffer::AutoCommandBufferBuilder;
//! use vulkano::command_buffer::CommandBuffer;
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! # let queue: std::sync::Arc<vulkano::device::Queue> = return;
//! let cb = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
//!     // TODO: add an actual command to this example
//!     .build().unwrap();
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

pub use self::auto::AutoCommandBuffer;
pub use self::auto::AutoCommandBufferBuilder;
pub use self::auto::AutoCommandBufferBuilderContextError;
pub use self::auto::BeginRenderPassError;
pub use self::auto::BlitImageError;
pub use self::auto::BuildError;
pub use self::auto::ClearColorImageError;
pub use self::auto::CopyBufferError;
pub use self::auto::CopyBufferImageError;
pub use self::auto::CopyImageError;
pub use self::auto::DebugMarkerError;
pub use self::auto::DispatchError;
pub use self::auto::DrawError;
pub use self::auto::DrawIndexedError;
pub use self::auto::DrawIndexedIndirectError;
pub use self::auto::DrawIndirectError;
pub use self::auto::ExecuteCommandsError;
pub use self::auto::FillBufferError;
pub use self::auto::UpdateBufferError;
pub use self::state_cacher::StateCacher;
pub use self::state_cacher::StateCacherOutcome;
pub use self::traits::CommandBuffer;
pub use self::traits::CommandBufferExecError;
pub use self::traits::CommandBufferExecFuture;

use framebuffer::{EmptySinglePassRenderPassDesc, Framebuffer, RenderPass, Subpass};
use pipeline::depth_stencil::DynamicStencilValue;
use pipeline::viewport::{Scissor, Viewport};
use query::QueryPipelineStatisticFlags;
use std::sync::Arc;

pub mod pool;
pub mod submit;
pub mod synced;
pub mod sys;
pub mod validity;

mod auto;
mod state_cacher;
mod traits;

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
#[repr(u32)]
pub enum SubpassContents {
    /// The subpass will only directly contain commands.
    Inline = vk::SUBPASS_CONTENTS_INLINE,
    /// The subpass will only contain secondary command buffers invocations.
    SecondaryCommandBuffers = vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS,
}

/// Determines the kind of command buffer that we want to create.
#[derive(Debug, Clone)]
pub enum Kind<R, F> {
    /// A primary command buffer can execute all commands and can call secondary command buffers.
    Primary,

    /// A secondary command buffer.
    Secondary {
        /// If `Some`, can only call draw operations that can be executed from within a specific
        /// subpass. Otherwise it can execute all dispatch and transfer operations, but not drawing
        /// operations.
        render_pass: Option<KindSecondaryRenderPass<R, F>>,

        /// Whether it is allowed to have an active occlusion query in the primary command buffer
        /// when executing this secondary command buffer.
        occlusion_query: KindOcclusionQuery,

        /// Which pipeline statistics queries are allowed to be active when this secondary command
        /// buffer starts.
        ///
        /// Note that the `pipeline_statistics_query` feature must be enabled if any of the flags
        /// of this value are set.
        query_statistics_flags: QueryPipelineStatisticFlags,
    },
}

/// Additional information for `Kind::Secondary`.
#[derive(Debug, Clone)]
pub struct KindSecondaryRenderPass<R, F> {
    /// Which subpass this secondary command buffer can be called from.
    pub subpass: Subpass<R>,

    /// The framebuffer object that will be used when calling the command buffer.
    /// This parameter is optional and is an optimization hint for the implementation.
    pub framebuffer: Option<F>,
}

/// Additional information for `Kind::Secondary`.
#[derive(Debug, Copy, Clone)]
pub enum KindOcclusionQuery {
    /// It is allowed to have an active occlusion query in the primary command buffer when
    /// executing this secondary command buffer.
    ///
    /// The `inherited_queries` feature must be enabled on the device for this to be a valid option.
    Allowed {
        /// The occlusion query can have the `control_precise` flag.
        control_precise_allowed: bool,
    },

    /// It is forbidden to have an active occlusion query.
    Forbidden,
}

impl
    Kind<
        RenderPass<EmptySinglePassRenderPassDesc>,
        Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>,
    >
{
    /// Equivalent to `Kind::Primary`.
    ///
    /// > **Note**: If you use `let kind = Kind::Primary;` in your code, you will probably get a
    /// > compilation error because the Rust compiler couldn't determine the template parameters
    /// > of `Kind`. To solve that problem in an easy way you can use this function instead.
    #[inline]
    pub fn primary() -> Kind<
        Arc<RenderPass<EmptySinglePassRenderPassDesc>>,
        Arc<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>>,
    > {
        Kind::Primary
    }

    /// Equivalent to `Kind::Secondary`.
    ///
    /// > **Note**: If you use `let kind = Kind::Secondary;` in your code, you will probably get a
    /// > compilation error because the Rust compiler couldn't determine the template parameters
    /// > of `Kind`. To solve that problem in an easy way you can use this function instead.
    #[inline]
    pub fn secondary(
        occlusion_query: KindOcclusionQuery,
        query_statistics_flags: QueryPipelineStatisticFlags,
    ) -> Kind<
        Arc<RenderPass<EmptySinglePassRenderPassDesc>>,
        Arc<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>>,
    > {
        Kind::Secondary {
            render_pass: None,
            occlusion_query,
            query_statistics_flags,
        }
    }
}
