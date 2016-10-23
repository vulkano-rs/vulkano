// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error::Error;
use std::iter;
use std::sync::Arc;

use buffer::TrackedBuffer;
use command_buffer::CommandBufferBuilder;
use command_buffer::DynamicState;
use command_buffer::pool::CommandPool;
use command_buffer::StatesManager;
use command_buffer::SubmitInfo;
use command_buffer::SubmitBuilder;
use command_buffer::Submit;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use descriptor::PipelineLayoutRef;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use device::Device;
use device::Queue;
use framebuffer::traits::TrackedFramebuffer;
use framebuffer::RenderPass;
use framebuffer::RenderPassClearValues;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use pipeline::vertex::Source;
use sync::Fence;
use vk;

pub use self::bind_pipeline::CmdBindPipeline;
pub use self::blit_image_unsynced::{BlitRegion, BlitRegionAspect};
pub use self::blit_image_unsynced::{CmdBlitImageUnsynced, CmdBlitImageUnsyncedError};
pub use self::copy_buffer_unsynced::{CmdCopyBufferUnsynced, CmdCopyBufferUnsyncedError};
pub use self::empty::PrimaryCb;
pub use self::empty::PrimaryCbBuilder;
pub use self::set_state::{CmdSetState};
pub use self::update_buffer_unsynced::{CmdUpdateBufferUnsynced, CmdUpdateBufferUnsyncedError};

mod bind_pipeline;
mod blit_image_unsynced;
mod copy_buffer_unsynced;
pub mod dispatch;
pub mod draw;
pub mod empty;
pub mod execute;
pub mod fill_buffer;
pub mod render_pass;
mod set_state;
pub mod update_buffer;
mod update_buffer_unsynced;

/// A list of commands that can be turned into a command buffer.
pub unsafe trait CommandsList {
    /// Adds a command that writes the content of a buffer.
    ///
    /// After this command is executed, the content of `buffer` will become `data`.
    #[inline]
    fn update_buffer<'a, B, D: ?Sized>(self, buffer: B, data: &'a D)
                                       -> update_buffer::UpdateCommand<'a, Self, B, D>
        where Self: Sized + CommandsListPossibleOutsideRenderPass, B: TrackedBuffer, D: Copy + 'static
    {
        update_buffer::UpdateCommand::new(self, buffer, data)
    }

    /// Adds a command that writes the content of a buffer.
    #[inline]
    fn fill_buffer<B>(self, buffer: B, data: u32) -> fill_buffer::FillCommand<Self, B>
        where Self: Sized + CommandsListPossibleOutsideRenderPass, B: TrackedBuffer
    {
        fill_buffer::FillCommand::new(self, buffer, data)
    }

    /// Adds a command that executes a secondary command buffer.
    ///
    /// When you create a command buffer, you have the possibility to create either a primary
    /// command buffer or a secondary command buffer. Secondary command buffers can't be executed
    /// directly, but can be executed from a primary command buffer.
    ///
    /// A secondary command buffer can't execute another secondary command buffer. The only way
    /// you can use `execute` is to make a primary command buffer call a secondary command buffer.
    #[inline]
    fn execute<Cb>(self, command_buffer: Cb) -> execute::ExecuteCommand<Cb, Self>
        where Self: Sized, Cb: CommandsListOutput       /* FIXME: */
    {
        execute::ExecuteCommand::new(self, command_buffer)
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
        where Self: Sized + CommandsList + CommandsListPossibleOutsideRenderPass, Pl: PipelineLayoutRef,
              S: TrackedDescriptorSetsCollection, Pc: 'a
    {
        dispatch::DispatchCommand::new(self, pipeline, sets, dimensions, push_constants)
    }

    /// Adds a command that starts a render pass.
    ///
    /// If `secondary` is true, then you will only be able to add secondary command buffers while
    /// you're inside the first subpass on the render pass. If `secondary` is false, you will only
    /// be able to add inline draw commands and not secondary command buffers.
    ///
    /// You must call this before you can add draw commands.
    #[inline]
    fn begin_render_pass<F, C>(self, framebuffer: F, secondary: bool, clear_values: C)
                               -> render_pass::BeginRenderPassCommand<Self, F::RenderPass, F>
        where Self: Sized + CommandsListPossibleOutsideRenderPass,
              F: TrackedFramebuffer, F::RenderPass: RenderPass + RenderPassClearValues<C>
    {
        render_pass::BeginRenderPassCommand::new(self, framebuffer, secondary, clear_values)
    }

    /// Adds a command that jumps to the next subpass of the current render pass.
    #[inline]
    fn next_subpass(self, secondary: bool) -> render_pass::NextSubpassCommand<Self>
        where Self: Sized + CommandsListPossibleInsideRenderPass
    {
        render_pass::NextSubpassCommand::new(self, secondary)
    }

    /// Adds a command that ends the current render pass.
    ///
    /// This must be called after you went through all the subpasses and before you can build
    /// the command buffer or add further commands.
    #[inline]
    fn end_render_pass(self) -> render_pass::EndRenderPassCommand<Self>
        where Self: Sized + CommandsListPossibleInsideRenderPass
    {
        render_pass::EndRenderPassCommand::new(self)
    }

    /// Adds a command that draws.
    ///
    /// Can only be used from inside a render pass.
    #[inline]
    fn draw<'a, Pv, Pl, Prp, S, Pc, V>(self, pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
                                       dynamic: &DynamicState, vertices: V, sets: S,
                                       push_constants: &'a Pc)
                                       -> draw::DrawCommand<'a, Self, V, Pv, Pl, Prp, S, Pc>
        where Self: Sized + CommandsList + CommandsListPossibleInsideRenderPass, Pl: PipelineLayoutRef,
              S: TrackedDescriptorSetsCollection, Pc: 'a, Pv: Source<V>
    {
        draw::DrawCommand::regular(self, pipeline, dynamic, vertices, sets, push_constants)
    }

    /// Turns the commands list into a command buffer that can be submitted.
    #[inline]
    fn build(self) -> CommandBuffer<Self::Output> where Self: Sized + CommandsListConcrete {
        CommandsListConcrete::build(self)
    }

    /// Builds a boxed command buffer object.
    #[inline]
    fn abstract_build(self) -> CommandBuffer where Self: Sized + CommandsListAbstract {
        CommandsListAbstract::abstract_build(self)
    }

    /// Appends this list of commands at the end of a command buffer in construction.
    fn append<'a>(&'a self, builder: CommandBufferBuilder<'a>) -> CommandBufferBuilder<'a> {
        // TODO: temporary body until all the impls implement this function
        unimplemented!()
    }

    /// Returns true if the command buffer can be built. This function should always return true,
    /// except when we're building a primary command buffer that is inside a render pass.
    // TODO: remove function
    fn buildable_state(&self) -> bool { unimplemented!() }

    /// Returns the number of commands in the commands list.
    ///
    /// Note that multiple actual commands may count for just 1.
    // TODO: remove function
    fn num_commands(&self) -> usize { unimplemented!() }

    /// Checks whether the command can be executed on the given queue family.
    // TODO: error type?
    // TODO: remove function
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()> { unimplemented!() }

    /// Extracts the object that contains the states of all the resources of the commands list.
    ///
    /// Panics if the states were already extracted.
    // TODO: remove function
    fn extract_states(&mut self) -> StatesManager { unimplemented!() }

    /// Returns true if the given compute pipeline is currently binded in the commands list.
    // TODO: better API?
    // TODO: remove function
    fn is_compute_pipeline_bound(&self, pipeline: vk::Pipeline) -> bool { unimplemented!() }

    /// Returns true if the given graphics pipeline is currently binded in the commands list.
    // TODO: better API?
    // TODO: remove function
    fn is_graphics_pipeline_bound(&self, pipeline: vk::Pipeline) -> bool { unimplemented!() }
}

unsafe impl CommandsList for Box<CommandsList> {
    #[inline]
    fn append<'a>(&'a self, builder: CommandBufferBuilder<'a>) -> CommandBufferBuilder<'a> {
        (**self).append(builder)
    }

    #[inline]
    fn buildable_state(&self) -> bool {
        (**self).buildable_state()
    }

    #[inline]
    fn num_commands(&self) -> usize {
        (**self).num_commands()
    }

    #[inline]
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()> {
        (**self).check_queue_validity(queue)
    }

    #[inline]
    fn extract_states(&mut self) -> StatesManager {
        (**self).extract_states()
    }

    #[inline]
    fn is_compute_pipeline_bound(&self, pipeline: vk::Pipeline) -> bool {
        (**self).is_compute_pipeline_bound(pipeline)
    }

    #[inline]
    fn is_graphics_pipeline_bound(&self, pipeline: vk::Pipeline) -> bool {
        (**self).is_graphics_pipeline_bound(pipeline)
    }
}

pub unsafe trait CommandsListConcrete: CommandsList {
    type Pool: CommandPool;
    /// The type of the command buffer that will be generated.
    type Output: CommandsListOutput;

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
    unsafe fn raw_build<I, F>(self, in_s: &mut StatesManager, out: &mut StatesManager,
                              additional_elements: F, barriers: I,
                              final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<Self::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>;

    /// Turns the commands list into a command buffer that can be submitted.
    // This function isn't inline because `raw_build` implementations usually are inline.
    fn build(mut self) -> CommandBuffer<Self::Output> where Self: Sized {
        assert!(self.buildable_state(), "Tried to build a command buffer still inside a \
                                         render pass");

        let mut states_in = self.extract_states();
        let mut states_out = StatesManager::new(); 

        let output = unsafe {
            self.raw_build(&mut states_in, &mut states_out, |_| {},
                           iter::empty(), PipelineBarrierBuilder::new())
        };

        CommandBuffer {
            states: states_out,
            commands: output,
        }
    }
}

pub unsafe trait CommandsListAbstract: CommandsList {
    /// Turns the commands list into a command buffer that can be submitted.
    fn abstract_build(self) -> CommandBuffer where Self: Sized;
}

unsafe impl<C> CommandsListAbstract for C
    where C: CommandsListConcrete, <C as CommandsListConcrete>::Output: 'static
{
    #[inline]
    fn abstract_build(self) -> CommandBuffer {
        let tmp = CommandsListConcrete::build(self);

        CommandBuffer {
            states: tmp.states,
            commands: Box::new(tmp.commands) as Box<_>,
        }
    }
}

/// Extension trait for both `CommandsList` and `CommandsListOutput` that indicates that we're
/// possibly outside a render pass.
///
/// In other words, if this trait is *not* implemented then we're guaranteed *not* to be outside
/// of a render pass. If it is implemented, then we maybe are but that's not sure.
pub unsafe trait CommandsListPossibleOutsideRenderPass {
    /// Returns `true` if we're outside a render pass.
    fn is_outside_render_pass(&self) -> bool;
}

/// Extension trait for both `CommandsList` and `CommandsListOutput` that indicates that we're
/// possibly inside a render pass.
///
/// In other words, if this trait is *not* implemented then we're guaranteed *not* to be inside
/// a render pass. If it is implemented, then we maybe are but that's not sure.
// TODO: make all return values optional, since we're possibly not in a render pass
pub unsafe trait CommandsListPossibleInsideRenderPass {
    type RenderPass: RenderPass;

    /// Returns the number of the subpass we're in. The value is 0-indexed, so immediately after
    /// calling `begin_render_pass` the value will be `0`.
    ///
    /// The value should always be strictly inferior to the number of subpasses in the render pass.
    fn current_subpass_num(&self) -> u32;

    /// If true, only secondary command buffers can be added inside the subpass. If false, only
    /// inline draw commands can be added.
    fn secondary_subpass(&self) -> bool;

    /// Returns the description of the render pass we're in.
    // TODO: return a trait object instead?
    fn render_pass(&self) -> &Self::RenderPass;

    //fn current_subpass(&self) -> Subpass<&Self::RenderPass>;
}

pub unsafe trait CommandsListOutput<S = StatesManager> {
    /// Returns the inner object.
    // TODO: crappy API
    fn inner(&self) -> vk::CommandBuffer;

    /// Returns the device this object belongs to.
    fn device(&self) -> &Arc<Device>;

    unsafe fn on_submit(&self, states: &S, queue: &Arc<Queue>,
                        fence: &mut FnMut() -> Arc<Fence>) -> SubmitInfo;
}

pub struct CommandBuffer<C = Box<CommandsListOutput>> {
    states: StatesManager,
    commands: C,
}

unsafe impl<C> Submit for CommandBuffer<C> where C: CommandsListOutput {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.commands.device()
    }

    #[inline]
    unsafe fn append_submission<'a>(&'a self, mut base: SubmitBuilder<'a>, queue: &Arc<Queue>)
                                    -> Result<SubmitBuilder<'a>, Box<Error>>
    {
        let mut fence = None;
        let infos = self.commands.on_submit(&self.states, queue, &mut || {
            match &mut fence {
                f @ &mut None => {
                    let fe = Fence::new(self.device().clone()); *f = Some(fe.clone()); fe
                },
                &mut Some(ref f) => f.clone()
            }
        });

        for (sem_wait, sem_stage) in infos.semaphores_wait {
            base = base.add_wait_semaphore(sem_wait, sem_stage);
        }

        if !infos.pre_pipeline_barrier.is_empty() {
            unimplemented!()
        }

        base = base.add_command_buffer_raw(self.commands.inner());

        if !infos.post_pipeline_barrier.is_empty() {
            unimplemented!()
        }

        for sem_signal in infos.semaphores_signal {
            base = base.add_signal_semaphore(sem_signal);
        }

        if let Some(fence) = fence {
            base = base.add_fence_signal(fence);
        }

        Ok(base)
    }
}
