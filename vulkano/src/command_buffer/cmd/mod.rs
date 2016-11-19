// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::Buffer;
use buffer::TypedBuffer;
use command_buffer::cb::CommandsListBuildPrimary;
use command_buffer::cb::CommandsListBuildPrimaryPool;
use command_buffer::DynamicState;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::SecondaryCommandBuffer;
use descriptor::PipelineLayoutRef;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use device::Device;
use framebuffer::traits::TrackedFramebuffer;
use framebuffer::RenderPass;
use framebuffer::RenderPassClearValues;
use image::Layout;
use image::TrackedImage;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::Source;
use sync::AccessFlagBits;
use sync::PipelineStages;
use OomError;

pub use self::begin_render_pass::CmdBeginRenderPass;
pub use self::bind_index_buffer::CmdBindIndexBuffer;
pub use self::bind_descriptor_sets::{CmdBindDescriptorSets, CmdBindDescriptorSetsError};
pub use self::bind_pipeline::CmdBindPipeline;
pub use self::bind_vertex_buffers::CmdBindVertexBuffers;
pub use self::blit_image_unsynced::{BlitRegion, BlitRegionAspect};
pub use self::blit_image_unsynced::{CmdBlitImageUnsynced, CmdBlitImageUnsyncedError};
pub use self::clear_attachments::CmdClearAttachments;
pub use self::copy_buffer::{CmdCopyBuffer, CmdCopyBufferError};
pub use self::dispatch::{CmdDispatch, CmdDispatchError};
pub use self::dispatch_indirect::{CmdDispatchIndirect, CmdDispatchIndirectError};
pub use self::draw::CmdDraw;
pub use self::draw_indexed::CmdDrawIndexed;
pub use self::empty::{empty, EmptyCommandsList};
pub use self::end_render_pass::{CmdEndRenderPass, CmdEndRenderPassError};
pub use self::execute::CmdExecuteCommands;
pub use self::fill_buffer::{CmdFillBuffer, CmdFillBufferError};
pub use self::join::CommandsListJoin;
pub use self::next_subpass::{CmdNextSubpass, CmdNextSubpassError};
pub use self::push_constants::{CmdPushConstants, CmdPushConstantsError};
pub use self::set_state::{CmdSetState};
pub use self::update_buffer::{CmdUpdateBuffer, CmdUpdateBufferError};

mod begin_render_pass;
mod bind_descriptor_sets;
mod bind_index_buffer;
mod bind_pipeline;
mod bind_vertex_buffers;
mod blit_image_unsynced;
mod clear_attachments;
mod copy_buffer;
mod dispatch;
mod dispatch_indirect;
mod draw;
mod draw_indexed;
mod empty;
mod end_render_pass;
mod execute;
mod fill_buffer;
mod join;
mod next_subpass;
mod push_constants;
mod set_state;
mod update_buffer;

/// A list of commands that can be turned into a command buffer.
///
/// This is just a naked list of commands. It holds buffers, images, etc. but the list of commands
/// itself is not a Vulkan object.
pub unsafe trait CommandsList {
    /// Adds a command that writes the content of a buffer.
    ///
    /// After this command is executed, the content of `buffer` will become `data`. If `data` is
    /// smaller than `buffer`, then only the beginning of `buffer` will be modified and the rest
    /// will be left untouched. If `buffer` is smaller than `data`, `buffer` will be entirely
    /// written and no error is generated.
    ///
    /// This command is limited to 64kB (65536 bytes) of data and should only be used for small
    /// amounts of data. For large amounts of data, you are encouraged to write the data to a
    /// buffer and use `copy_buffer` instead.
    #[inline]
    fn update_buffer<'a, B, D: ?Sized>(self, buffer: B, data: &'a D)
                                       -> Result<CmdUpdateBuffer<'a, Self, B, D>, CmdUpdateBufferError>
        where Self: Sized + CommandsListPossibleOutsideRenderPass, B: Buffer, D: Copy + 'static
    {
        CmdUpdateBuffer::new(self, buffer, data)
    }

    /// Adds a command that copies the content of a buffer to another.
    ///
    /// If `source` is smaller than `destination`, only the beginning of `destination` will be
    /// modified.
    #[inline]
    fn copy_buffer<S, D>(self, source: S, destination: D)
                         -> Result<CmdCopyBuffer<Self, S, D>, CmdCopyBufferError>
        where Self: Sized + CommandsListPossibleOutsideRenderPass,
              S: Buffer, D: Buffer
    {
        CmdCopyBuffer::new(self, source, destination)
    }

    /// Adds a command that writes the content of a buffer.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatidely written through the entire buffer.
    ///
    /// > **Note**: This function is technically safe because buffers can only contain integers or
    /// > floating point numbers, which are always valid whatever their memory representation is.
    /// > But unless your buffer actually contains only 32-bits integers, you are encouraged to use
    /// > this function only for zeroing the content of a buffer by passing `0` for the data.
    #[inline]
    fn fill_buffer<B>(self, buffer: B, data: u32)
                      -> Result<CmdFillBuffer<Self, B>, CmdFillBufferError>
        where Self: Sized + CommandsListPossibleOutsideRenderPass, B: Buffer
    {
        CmdFillBuffer::new(self, buffer, data)
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
    fn execute_commands<Cb>(self, command_buffer: Cb) -> CmdExecuteCommands<Cb, Self>
        where Self: Sized, Cb: SecondaryCommandBuffer
    {
        CmdExecuteCommands::new(self, command_buffer)
    }

    /// Adds a command that executes a compute shader.
    ///
    /// The `dimensions` are the number of working groups to start. The GPU will execute the
    /// compute shader `dimensions[0] * dimensions[1] * dimensions[2]` times.
    ///
    /// The `pipeline` is the compute pipeline that will be executed, and the sets and push
    /// constants will be accessible to all the invocations.
    #[inline]
    fn dispatch<Pl, S, Pc>(self, pipeline: Arc<ComputePipeline<Pl>>, sets: S,
                           dimensions: [u32; 3], push_constants: Pc)
                           -> Result<CmdDispatch<Self, Pl, S, Pc>, CmdDispatchError>
        where Self: Sized + CommandsList, Pl: PipelineLayoutRef,
              S: TrackedDescriptorSetsCollection
    {
        CmdDispatch::new(self, pipeline, sets, dimensions, push_constants)
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
                               -> CmdBeginRenderPass<Self, F::RenderPass, F>
        where Self: Sized, F: TrackedFramebuffer,
              F::RenderPass: RenderPass + RenderPassClearValues<C>
    {
        CmdBeginRenderPass::new(self, framebuffer, secondary, clear_values)
    }

    /// Adds a command that jumps to the next subpass of the current render pass.
    #[inline]
    fn next_subpass(self, secondary: bool) -> Result<CmdNextSubpass<Self>, CmdNextSubpassError>
        where Self: Sized
    {
        CmdNextSubpass::new(self, secondary)
    }

    /// Adds a command that ends the current render pass.
    ///
    /// This must be called after you went through all the subpasses and before you can build
    /// the command buffer or add further commands.
    #[inline]
    fn end_render_pass(self) -> Result<CmdEndRenderPass<Self>, CmdEndRenderPassError>
        where Self: Sized
    {
        CmdEndRenderPass::new(self)
    }

    /// Adds a command that draws.
    ///
    /// Can only be used from inside a render pass.
    #[inline]
    fn draw<Pv, Pl, Prp, S, Pc, V>(self, pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
                                   dynamic: DynamicState, vertices: V, sets: S,
                                   push_constants: Pc)
                                   -> CmdDraw<Self, V, Pv, Pl, Prp, S, Pc>
        where Self: Sized + CommandsList,
              Pl: PipelineLayoutRef,
              S: TrackedDescriptorSetsCollection,
              Pv: Source<V>
    {
        CmdDraw::new(self, pipeline, dynamic, vertices, sets, push_constants)
    }

    /// Adds a command that draws with an index buffer.
    ///
    /// Can only be used from inside a render pass.
    #[inline]
    fn draw_indexed<Pv, Pl, Prp, S, Pc, V, Ib, I>(self, pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
                                               dynamic: DynamicState, vertices: V, indices: Ib,
                                               sets: S, push_constants: Pc)
                                               -> CmdDrawIndexed<Self, V, Ib, Pv, Pl, Prp, S, Pc>
        where Self: Sized + CommandsList,
              Pl: PipelineLayoutRef,
              S: TrackedDescriptorSetsCollection,
              Pv: Source<V>,
              Ib: Buffer + TypedBuffer<Content = [I]>,
              I: Index + 'static
    {
        CmdDrawIndexed::new(self, pipeline, dynamic, vertices, indices, sets, push_constants)
    }

    /// Appends another list at the end of this one.
    #[inline]
    fn join<L>(self, other: L) -> CommandsListJoin<Self, L> where Self: Sized, L: CommandsList {
        CommandsListJoin::new(self, other)
    }

    /// Builds the list as a primary command buffer.
    #[inline]
    fn build_primary<C>(self, device: &Arc<Device>, queue_family: QueueFamily)
                        -> Result<C, OomError>
        where C: CommandsListBuildPrimary<Self>, Self: Sized
    {
        CommandsListBuildPrimary::build_primary(device, queue_family, self)
    }

    /// Builds the list as a primary command buffer and with the given pool.
    #[inline]
    fn build_primary_with_pool<C, P>(self, pool: P) -> Result<C, OomError>
        where C: CommandsListBuildPrimaryPool<Self, P>, Self: Sized
    {
        CommandsListBuildPrimaryPool::build_primary_with_pool(pool, self)
    }

    /// Appends this list of commands at the end of a command buffer in construction.
    ///
    /// The `CommandsListSink` typically represents a command buffer being constructed.
    /// The `append` method must call the methods of that `CommandsListSink` in order to add
    /// elements at the end of the command buffer being constructed. The `CommandsListSink` can
    /// also typically be a filter around another `CommandsListSink`.
    ///
    /// The lifetime of the `CommandsListSink` is the same as the lifetime of `&self`. This means
    /// that the commands you pass to the sink can borrow `self`.
    ///
    /// # Safety
    ///
    /// It is important for safety that `append` always returns the same commands.
    ///
    /// > **Note**: For example, in the case secondary command buffers this function is called once
    /// > when the secondary command buffer is created, and once again every time the secondary
    /// > command buffer is used. All the calls must match in order for the behavior to be safe.
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>);
}

unsafe impl CommandsList for Box<CommandsList> {
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        (**self).append(builder)
    }
}

/// Output of the "append" method. The lifetime corresponds to a borrow of the commands list.
///
/// A `CommandsListSink` typically represents a command buffer being constructed. The various
/// methods add elements at the end of that command buffer.
pub trait CommandsListSink<'a> {
    /// Returns the device of the sink. Used by the commands in the commands list to make sure that
    /// their buffer, images, etc. belong to the same device as the sink.
    fn device(&self) -> &Arc<Device>;

    /// Requests that a command must be executed.
    ///
    /// Note that the lifetime means that we hold a reference to the content of
    /// the commands list in that closure.
    fn add_command(&mut self, Box<CommandsListSinkCaller<'a> + 'a>);

    /// Requests that a buffer must be transitionned to a given state.
    ///
    /// The parameters are the buffer, and its offset and size, plus a `write` boolean that is
    /// `true` if the buffer must be transitionned to a writable state or `false` if it must be
    /// transitionned to a readable state.
    fn add_buffer_transition(&mut self, buffer: &Buffer, offset: usize, size: usize,
                             write: bool, stages: PipelineStages, access: AccessFlagBits);

    /// Requests that an image must be transitionned to a given state.
    ///
    /// If necessary, you must transition the image to the `layout`.
    fn add_image_transition(&mut self, image: &TrackedImage, first_layer: u32, num_layers: u32,
                            first_mipmap: u32, num_mipmaps: u32, write: bool, layout: Layout,
                            stages: PipelineStages, access: AccessFlagBits);

    /// Notifies the sink that an image has been transitionned by one of the previous commands
    /// added with `add_command`.
    ///
    /// The sink doesn't need to perform any operation when this method is called, but should
    /// modify its internal state in order to keep track of the state of that image.
    fn add_image_transition_notification(&mut self, image: &TrackedImage, first_layer: u32,
                                         num_layers: u32, first_mipmap: u32, num_mipmaps: u32,
                                         layout: Layout, stages: PipelineStages,
                                         access: AccessFlagBits);
}

/// This trait is equivalent to `FnOnce(&mut RawCommandBufferPrototype<'a>)`. It is necessary
/// because Rust doesn't permit you to call a `Box<FnOnce>`.
///
/// > **Note**: This trait will most likely be removed if Rust fixes that problem with
/// > `Box<FnOnce>`.
pub trait CommandsListSinkCaller<'a> {
    /// Consumes a `Box<CommandsListSinkCaller>` and call it on the parameter.
    fn call(self: Box<Self>, &mut RawCommandBufferPrototype<'a>);
}

impl<'a, T> CommandsListSinkCaller<'a> for T
    where T: FnOnce(&mut RawCommandBufferPrototype<'a>) -> () + 'a
{
    fn call(self: Box<Self>, proto: &mut RawCommandBufferPrototype<'a>) {
        self(proto);
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
