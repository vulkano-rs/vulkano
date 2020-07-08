// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Contains `SyncCommandBufferBuilder` and `SyncCommandBuffer`.

pub use self::base::SyncCommandBuffer;
pub use self::base::SyncCommandBufferBuilder;
pub use self::base::SyncCommandBufferBuilderError;
pub use self::commands::SyncCommandBufferBuilderBindDescriptorSets;
pub use self::commands::SyncCommandBufferBuilderBindVertexBuffer;
pub use self::commands::SyncCommandBufferBuilderExecuteCommands;

mod base;
mod commands;

#[cfg(test)]
mod tests;
