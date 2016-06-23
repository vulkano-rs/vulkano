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

pub use self::inner::Submission;
pub use self::outer::submit;
pub use self::outer::DynamicState;
pub use self::outer::PrimaryCommandBufferBuilder;
pub use self::outer::PrimaryCommandBufferBuilderInlineDraw;
pub use self::outer::PrimaryCommandBufferBuilderSecondaryDraw;
pub use self::outer::PrimaryCommandBuffer;
pub use self::outer::SecondaryGraphicsCommandBufferBuilder;
pub use self::outer::SecondaryGraphicsCommandBuffer;
pub use self::outer::SecondaryComputeCommandBufferBuilder;
pub use self::outer::SecondaryComputeCommandBuffer;

use std::sync::Arc;

use device::Device;
use self::pool::CommandPool;
use self::sys::UnsafeCommandBuffer;

mod inner;
mod outer;
pub mod pool;
pub mod sys;

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

/// Trait for types that contain a list of commands executable by a queue.
pub unsafe trait CommandBuffer {
    /// The pool that was used to create this command buffer.
    type Pool: CommandPool;

    /// Returns the inner command buffer.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    fn inner_cb(&self) -> &UnsafeCommandBuffer<Self::Pool>;

    /// Returns the pool used to create this command buffer.
    #[inline]
    fn pool(&self) -> &Self::Pool {
        self.inner_cb().pool()
    }

    /// Returns the device this command buffer belongs to.
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner_cb().device()
    }

    /// Returns true if the command buffer was created with the `SimultaneousUse` flag.
    ///
    /// If false, then the command buffer must be properly synchronized.
    #[inline]
    fn simultaneous_use(&self) -> bool {
        self.inner_cb().simultaneous_use()
    }

    /// Returns true if this is a secondary command buffer.
    #[inline]
    fn is_secondary(&self) -> bool {
        self.inner_cb().is_secondary()
    }
}
