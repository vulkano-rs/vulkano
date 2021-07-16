// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Contains `SyncCommandBufferBuilder` and `SyncCommandBuffer`.
//!
//! # How pipeline stages work in Vulkan
//!
//! Imagine you create a command buffer that contains 10 dispatch commands, and submit that command
//! buffer. According to the Vulkan specs, the implementation is free to execute the 10 commands
//! simultaneously.
//!
//! Now imagine that the command buffer contains 10 draw commands instead. Contrary to the dispatch
//! commands, the draw pipeline contains multiple stages: draw indirect, vertex input, vertex shader,
//! ..., fragment shader, late fragment test, color output. When there are multiple stages, the
//! implementations must start and end the stages in order. In other words it can start the draw
//! indirect stage of all 10 commands, then start the vertex input stage of all 10 commands, and so
//! on. But it can't for example start the fragment shader stage of a command before starting the
//! vertex shader stage of another command. Same thing for ending the stages in the right order.
//!
//! Depending on the type of the command, the pipeline stages are different. Compute shaders use the
//! compute stage, while transfer commands use the transfer stage. The compute and transfer stages
//! aren't ordered.
//!
//! When you submit multiple command buffers to a queue, the implementation doesn't do anything in
//! particular and behaves as if the command buffers were appended to one another. Therefore if you
//! submit a command buffer with 10 dispatch commands, followed with another command buffer with 5
//! dispatch commands, then the implementation can perform the 15 commands simultaneously.
//!
//! ## Introducing barriers
//!
//! In some situations this is not the desired behaviour. If you add a command that writes to a
//! buffer followed with another command that reads that buffer, you don't want them to execute
//! simultaneously. Instead you want the second one to wait until the first one is finished. This
//! is done by adding a pipeline barrier between the two commands.
//!
//! A pipeline barriers has a source stage and a destination stage (plus various other things).
//! A barrier represents a split in the list of commands. When you add it, the stages of the commands
//! before the barrier corresponding to the source stage of the barrier, must finish before the
//! stages of the commands after the barrier corresponding to the destination stage of the barrier
//! can start.
//!
//! For example if you add a barrier that transitions from the compute stage to the compute stage,
//! then the compute stage of all the commands before the barrier must end before the compute stage
//! of all the commands after the barrier can start. This is appropriate for the example about
//! writing then reading the same buffer.
//!
//! ## Batching barriers
//!
//! Since barriers are "expensive" (as the queue must block), vulkano attempts to group as many
//! pipeline barriers as possible into one.
//!
//! Adding a command to a sync command buffer builder does not immediately add it to the underlying
//! command buffer builder. Instead the command is added to a queue, and the builder keeps a
//! prototype of a barrier that must be added before the commands in the queue are flushed.
//!
//! Whenever you add a command, the builder will find out whether a barrier is needed before the
//! command. If so, it will try to merge this barrier with the prototype and add the command to the
//! queue. If not possible, the queue will be entirely flushed and the command added to a fresh new
//! queue with a fresh new barrier prototype.

pub use self::base::SyncCommandBuffer;
pub use self::base::SyncCommandBufferBuilder;
pub use self::base::SyncCommandBufferBuilderError;
pub use self::commands::SyncCommandBufferBuilderBindDescriptorSets;
pub use self::commands::SyncCommandBufferBuilderBindVertexBuffer;
pub use self::commands::SyncCommandBufferBuilderExecuteCommands;

mod base;
mod commands;
