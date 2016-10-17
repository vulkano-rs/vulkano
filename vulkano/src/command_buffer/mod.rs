// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Commands that the GPU will execute (includes draw commands).
//!
//! With Vulkan, before the GPU can do anything you must create a `CommandBuffer`. A command buffer
//! is a list of commands that will executed by the GPU. Once a command buffer is created, you can
//! execute it. A command buffer must be created even for the most simple tasks.
//!
//! # Pools
//!
//! Command buffers are allocated from pools. You must first create a command buffer pool which
//! you will create command buffers from.
//!
//! A pool is linked to a queue family. Command buffers that are created from a certain pool can
//! only be submitted to queues that belong to that specific family.
//!
//! # Primary and secondary command buffers.
//!
//! There are three types of command buffers:
//!
//! - **Primary command buffers**. They can contain any command. They are the only type of command
//!   buffer that can be submitted to a queue.
//! - **Secondary "graphics" command buffers**. They contain draw and clear commands. They can be
//!   called from a primary command buffer once a framebuffer has been selected.
//! - **Secondary "compute" command buffers**. They can contain non-draw and non-clear commands
//!   (eg. copying between buffers) and can be called from a primary command buffer outside of a
//!   render pass.
//!
//! Note that secondary command buffers cannot call other command buffers.
//!

// Implementation note.
// There are various restrictions about which command can be used at which moment. Therefore the
// API has several different command buffer wrappers, but they all use the same internal
// struct. The restrictions are enforced only in the public types.

pub use self::states_manager::StatesManager;
pub use self::cmd::PrimaryCbBuilder;
pub use self::cmd::CommandBuffer;
pub use self::cmd::CommandsList;
pub use self::submit::Submission;
pub use self::submit::Submit;
pub use self::submit::SubmitBuilder;
pub use self::submit::SubmitChain;

use std::sync::Arc;
use command_buffer::sys::PipelineBarrierBuilder;
use pipeline::viewport::Viewport;
use pipeline::viewport::Scissor;
use sync::PipelineStages;
use sync::Semaphore;

pub mod cmd;
pub mod pool;
pub mod sys;

mod states_manager;
mod submit;

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

/// The dynamic state to use for a draw command.
#[derive(Debug, Clone)]
pub struct DynamicState {
    pub line_width: Option<f32>,
    pub viewports: Option<Vec<Viewport>>,
    pub scissors: Option<Vec<Scissor>>,
}

impl DynamicState {
    #[inline]
    pub fn none() -> DynamicState {
        DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
        }
    }
}

impl Default for DynamicState {
    #[inline]
    fn default() -> DynamicState {
        DynamicState::none()
    }
}

/// Information about how the submitting function should synchronize the submission.
// TODO: rework that design? move to std?
pub struct SubmitInfo {
    /// List of semaphores to wait upon before the command buffer starts execution.
    pub semaphores_wait: Vec<(Arc<Semaphore>, PipelineStages)>,
    /// List of semaphores to signal after the command buffer has finished.
    pub semaphores_signal: Vec<Arc<Semaphore>>,
    /// Pipeline barrier to execute on the queue and immediately before the command buffer.
    /// Ignored if empty.
    pub pre_pipeline_barrier: PipelineBarrierBuilder,
    /// Pipeline barrier to execute on the queue and immediately after the command buffer.
    /// Ignored if empty.
    pub post_pipeline_barrier: PipelineBarrierBuilder,
}

impl SubmitInfo {
    #[inline]
    pub fn empty() -> SubmitInfo {
        SubmitInfo {
            semaphores_wait: Vec::new(),
            semaphores_signal: Vec::new(),
            pre_pipeline_barrier: PipelineBarrierBuilder::new(),
            post_pipeline_barrier: PipelineBarrierBuilder::new(),
        }
    }
}
