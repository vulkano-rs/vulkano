// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Standard implementation of the `CommandBuffer` trait.
//! 
//! Everything in this module is dedicated to the "standard" implementation of command buffers.

use std::ops::Range;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::CommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use image::Image;
use image::sys::Layout;
use sync::AccessFlagBits;
use sync::PipelineStages;

use self::update_buffer::StdUpdateBufferBuilder;

pub mod primary;
pub mod update_buffer;

mod sync_helper;

///
pub unsafe trait StdCommandBufferBuilder {
    /// The finished command buffer.
    type BuildOutput: CommandBuffer;

    /// The command pool that was used to build the command buffer.
    type Pool: CommandPool;

    /// Adds a buffer update command at the end of the command buffer builder.
    #[inline]
    fn update_buffer<'a, 'b, D, S, B: 'b>(self, buffer: S, data: &'a D)
                                          -> StdUpdateBufferBuilder<'a, Self, D, B>
        where Self: Sized,
              B: Buffer,
              S: Into<BufferSlice<'b, D, B>>,
              D: Copy + 'static,
    {
        StdUpdateBufferBuilder::new(self, buffer, data)
    }

    unsafe fn add_command<F>(&mut self, cmd: F)
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<Self::Pool>);

    unsafe fn add_buffer_usage<B>(&mut self, buffer: &Arc<B>, slice: Range<usize>, write: bool,
                                  stages: PipelineStages, accesses: AccessFlagBits)
        where B: Buffer;

    unsafe fn add_image_usage<I>(&mut self, image: &Arc<I>, mipmaps: Range<u32>,
                                 array_layers: Range<u32>, write: bool, layout: Layout,
                                 stages: PipelineStages, accesses: AccessFlagBits)
        where I: Image;

    /// Returns true if the parameter is the same pipeline as the one that is currently binded on
    /// the graphics slot.
    ///
    /// Since this is purely an optimization to avoid having to bind the pipeline again, you can
    /// return `false` when in doubt.
    ///
    /// This function doesn't take into account any possible command that you add through
    /// `add_command`.
    #[inline]
    fn is_current_graphics_pipeline(&self /*, pipeline: &P */) -> bool {
        false
    }

    /// Finishes building the command buffer.
    ///
    /// Consumes the builder and returns an implementation of `StdCommandBuffer`.
    fn build(self) -> Self::BuildOutput;
}
