// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Internals of vulkano's command buffers building.
//!
//! You probably don't need to look inside this module if you're a beginner. The
//! `AutoCommandBufferBuilder` provided in the parent module should be good for most needs.
//!
//! # Builder basics
//!
//! The lowest-level command buffer types are `UnsafeCommandBufferBuilder` and
//! `UnsafeCommandBuffer`. These two types have zero overhead over Vulkan command buffers but are
//! very unsafe to use.
//!
//! Before you can add a command to an unsafe command buffer builder, you should:
//!
//! - Make sure that the buffers or images used by the command stay alive for the duration of the
//!   command buffer.
//! - Check that the device used by the buffers or images of the command is the same as the device
//!   of the command buffer.
//! - If the command buffer is inside/outside a render pass, check that the command can be executed
//!   inside/outside a render pass. Same for secondary command buffers.
//! - Check that the command can be executed on the queue family of the command buffer. Some queue
//!   families don't support graphics and/or compute operations .
//! - Make sure that when the command buffer is submitted the buffers and images of the command
//!   will be properly synchronized.
//! - Make sure that pipeline barriers are correctly inserted in order to avoid race conditions.
//!
//! In order to allow you to customize which checks are performed, vulkano provides *layers*. They
//! are structs that can be put around a command buffer builder and that will perform them. Keep
//! in mind that all the conditions above must be respected, but if you somehow make sure at
//! compile-time that some requirements are always correct, you can avoid paying some runtime cost
//! by not using all layers.
//!
//! Adding a command to a command buffer builder is done in two steps:
//!
//! - First you must build a struct that represents the command to add. The struct's constructor
//!   can perform various checks to make sure that the command itself is valid, or it can provide
//!   an unsafe constructor that doesn't perform any check.
//! - Then use the `AddCommand` trait to add it. The trait is implemented on the command buffer
//!   builder and on the various layers, and its template parameter is the struct representing
//!   the command.
//!
//! Since the `UnsafeCommandBufferBuilder` doesn't keep the command structs alive (as it would
//! incur an overhead), it implements `AddCommand<&T>`.
//!
//! The role of the `CommandsListLayer` and `BufferedCommandsListLayer` layers is to keep the
//! commands alive. They implement `AddCommand<T>` if the builder they wrap around implements
//! `AddCommand<&T>`. In other words they are the lowest level that you should put around an
//! `UnsafeCommandBufferBuilder`.
//!
//! The other layers of this module implement `AddCommand<T>` if the builder they wrap around
//! implements `AddCommand<T>`.
//!
//! # Building a command buffer
//!
//! Once you are satisfied with the commands you added to a builder, use the `CommandBufferBuild`
//! trait to build it.
//!
//! This trait is implemented on the `UnsafeCommandBufferBuilder` but also on all the layers.
//! The builder's layers can choose to add layers around the finished command buffer.
//!
//! # The `CommandsList` trait
//!
//! The `CommandsList` trait is implemented on any command buffer or command buffer builder that
//! exposes a list of commands. It is required by some of the layers.

pub use self::auto_barriers::AutoPipelineBarriersLayer;
pub use self::buffered::BufferedCommandsListLayer;
pub use self::buffered::BufferedCommandsListLayerCommands;
pub use self::commands_list::CommandsList;
pub use self::commands_list::CommandsListLayer;
pub use self::context_check::ContextCheckLayer;
pub use self::device_check::DeviceCheckLayer;
pub use self::queue_ty_check::QueueTyCheckLayer;
pub use self::state_cache::StateCacheLayer;
pub use self::sys::Kind;
pub use self::sys::Flags;
pub use self::sys::UnsafeCommandBufferBuilder;
pub use self::sys::UnsafeCommandBuffer;
pub use self::traits::AddCommand;
pub use self::traits::CommandBufferBuild;

mod auto_barriers;
mod buffered;
mod commands_list;
mod device_check;
mod context_check;
mod queue_ty_check;
mod state_cache;
mod sys;
mod traits;
