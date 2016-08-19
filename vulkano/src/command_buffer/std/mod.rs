// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;
use std::sync::Arc;

use buffer::traits::TrackedBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::submit::CommandBuffer;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use descriptor::PipelineLayout;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use framebuffer::Framebuffer;
use framebuffer::RenderPass;
use image::traits::TrackedImage;
use instance::QueueFamily;
use pipeline::ComputePipeline;

pub use self::empty::PrimaryCb;
pub use self::empty::PrimaryCbBuilder;

pub mod dispatch;
pub mod empty;
pub mod render_pass;
pub mod update_buffer;

/// A list of commands that can be turned into a command buffer.
pub unsafe trait StdCommandsList {
    /// The type of the pool that will be used to create the command buffer.
    type Pool: CommandPool;
    /// The type of the command buffer that will be generated.
    type Output: CommandBuffer<Pool = Self::Pool>;

    /// Adds a command that writes the content of a buffer.
    ///
    /// After this command is executed, the content of `buffer` will become `data`.
    #[inline]
    fn update_buffer<'a, B, D: ?Sized>(self, buffer: B, data: &'a D)
                                       -> update_buffer::UpdateCommand<'a, Self, B, D>
        where Self: Sized + OutsideRenderPass, B: TrackedBuffer, D: Copy + 'static
    {
        update_buffer::UpdateCommand::new(self, buffer, data)
    }

    /// Adds a command that executes a compute shader.
    ///
    /// The `dimensions` are the number of working groups to start. The GPU will execute the
    /// compute shader `dimensions[0] * dimensions[1] * dimensions[2]` times.
    ///
    /// The `pipeline` is the compute pipeline that will be executed, and the sets and push
    /// constants will be accessible to all the invocations.
    #[inline]
    fn dispatch<'a, Pl, S, Pc>(self, pipeline: Arc<ComputePipeline<Pl>>, sets: S,
                               dimensions: [u32; 3], push_constants: &'a Pc)
                               -> dispatch::DispatchCommand<'a, Self, Pl, S, Pc>
        where Self: Sized + StdCommandsList + OutsideRenderPass, Pl: PipelineLayout,
              S: TrackedDescriptorSetsCollection, Pc: 'a
    {
        dispatch::DispatchCommand::new(self, pipeline, sets, dimensions, push_constants)
    }

    /// Adds a command that starts a render pass.
    ///
    /// You must call this before you can add draw commands.
    #[inline]
    fn begin_render_pass<Rp>(self, framebuffer: Arc<Framebuffer<Rp>>)
                             -> render_pass::BeginRenderPassCommand<Self, Rp, Rp>
        where Self: Sized + OutsideRenderPass, Rp: RenderPass
    {
        render_pass::BeginRenderPassCommand::new(self, framebuffer)
    }

    /// Adds a command that jumps to the next subpass of the current render pass.
    fn next_subpass(self, secondary: bool) -> render_pass::NextSubpassCommand<Self>
        where Self: Sized + InsideRenderPass
    {
        render_pass::NextSubpassCommand::new(self, secondary)
    }

    /// Adds a command that ends the current render pass.
    ///
    /// This must be called after you went through all the subpasses and before you can build
    /// the command buffer or add further commands.
    fn end_render_pass(self) -> render_pass::EndRenderPassCommand<Self>
        where Self: Sized + InsideRenderPass
    {
        render_pass::EndRenderPassCommand::new(self)
    }

    /// Returns true if the command buffer can be built. This function should always return true,
    /// except when we're building a primary command buffer that is inside a render pass.
    fn buildable_state(&self) -> bool;

    /// Turns the commands list into a command buffer that can be submitted.
    fn build(self) -> Self::Output where Self: Sized {
        assert!(self.buildable_state(), "Tried to build a command buffer still inside a \
                                         render pass");

        unsafe {
            self.raw_build(|_| {}, iter::empty(), PipelineBarrierBuilder::new())
        }
    }

    /// Returns the number of commands in the commands list.
    ///
    /// Note that multiple actual commands may count for just 1.
    fn num_commands(&self) -> usize;

    /// Checks whether the command can be executed on the given queue family.
    // TODO: error type?
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()>;

    /// Returns the current status of a buffer, or `None` if the buffer hasn't been used yet.
    ///
    /// Whether the buffer passed as parameter is the same as the one in the commands list must be
    /// determined with the `is_same` method of `TrackedBuffer`.
    ///
    /// Calling this function tells the commands list that you are going to manage the
    /// synchronization that buffer yourself. Hence why the function is unsafe.
    ///
    /// This function is not meant to be called, except when writing a wrapper around a
    /// commands list.
    ///
    /// # Panic
    ///
    /// - Panics if the state of that buffer has already been previously extracted.
    ///
    unsafe fn extract_current_buffer_state<B>(&mut self, buffer: &B) -> Option<B::CommandListState>
        where B: TrackedBuffer;

    /// Returns the current status of an image, or `None` if the image hasn't been used yet.
    ///
    /// See the description of `extract_current_buffer_state`.
    ///
    /// # Panic
    ///
    /// - Panics if the state of that image has already been previously extracted.
    ///
    unsafe fn extract_current_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage;

    /// Turns the commands list into a command buffer.
    ///
    /// This function accepts additional arguments that will customize the output:
    ///
    /// - `additional_elements` is a closure that must be called on the command buffer builder
    ///   after it has finished building and before `final_barrier` are added.
    /// - `barriers` is a list of pipeline barriers accompanied by a command number. The
    ///   pipeline barrier must happen after the given command number. Usually you want all the
    ///   the command numbers to be inferior to `num_commands`.
    /// - `final_barrier` is a pipeline barrier that must be added at the end of the
    ///   command buffer builder.
    ///
    /// This function doesn't check that `buildable_state` returns true.
    unsafe fn raw_build<I, F>(self, additional_elements: F, barriers: I,
                              final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<Self::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>;
}

/// Extension trait for `StdCommandsList` that indicates that we're outside a render pass.
pub unsafe trait OutsideRenderPass: StdCommandsList {}

/// Extension trait for `StdCommandsList` that indicates that we're inside a render pass.
pub unsafe trait InsideRenderPass: StdCommandsList {
    type RenderPass: RenderPass;
    type Framebuffer;

    /// Returns the number of the subpass we're in. The value is 0-indexed, so immediately after
    /// calling `begin_render_pass` the value will be `0`.
    ///
    /// The value should always be strictly inferior to the number of subpasses in the render pass.
    fn current_subpass(&self) -> u32;

    // TODO: don't use Arc
    fn render_pass(&self) -> &Arc<Self::RenderPass>;

    fn framebuffer(&self) -> &Self::Framebuffer;
}
