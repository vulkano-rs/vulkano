// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use buffer::BufferAccess;
use command_buffer::pool::UnsafeCommandPoolAlloc;
use command_buffer::sys::Flags;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use command_buffer::sys::UnsafeCommandBufferBuilderPipelineBarrier;
use command_buffer::CommandBufferExecError;
use command_buffer::Kind;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use fnv::FnvHashMap;
use framebuffer::FramebufferAbstract;
use framebuffer::RenderPassAbstract;
use image::ImageAccess;
use image::ImageLayout;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::Mutex;
use sync::AccessCheckError;
use sync::AccessError;
use sync::AccessFlagBits;
use sync::GpuFuture;
use sync::PipelineMemoryAccess;
use sync::PipelineStages;
use OomError;

/// Wrapper around `UnsafeCommandBufferBuilder` that handles synchronization for you.
///
/// Each method of the `UnsafeCommandBufferBuilder` has an equivalent in this wrapper, except
/// for `pipeline_layout` which is automatically handled. This wrapper automatically builds
/// pipeline barriers, keeps used resources alive and implements the `CommandBuffer` trait.
///
/// Since the implementation needs to cache commands in a `Vec`, most methods have additional
/// `Send + Sync + 'static` trait requirements on their generics.
///
/// If this builder finds out that a command isn't valid because of synchronization reasons (eg.
/// trying to copy from a buffer to an image which share the same memory), then an error is
/// returned.
/// Note that all methods are still unsafe, because this builder doesn't check the validity of
/// the commands except for synchronization purposes. The builder may panic if you pass invalid
/// commands.
pub struct SyncCommandBufferBuilder {
    // The actual Vulkan command buffer builder.
    inner: UnsafeCommandBufferBuilder,

    // Resources and their accesses. Used for executing secondary command buffers in a primary.
    buffers: Vec<(ResourceLocation, PipelineMemoryAccess)>,
    images: Vec<(
        ResourceLocation,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
    )>,

    // Stores the current state of all resources (buffers and images) that are in use by the
    // command buffer.
    resources: FnvHashMap<BuilderKey, ResourceState>,

    // Prototype for the pipeline barrier that must be submitted before flushing the commands
    // in `commands`.
    pending_barrier: UnsafeCommandBufferBuilderPipelineBarrier,

    // Stores all the commands that were added to the sync builder. Some of them are maybe not
    // submitted to the inner builder yet. A copy of this `Arc` is stored in each `BuilderKey`.
    commands: Arc<Mutex<Commands>>,

    // Locations within commands that pipeline barriers were inserted. For debugging purposes.
    // TODO: present only in cfg(debug_assertions)?
    barriers: Vec<usize>,

    // `true` if the builder has been put in an inconsistent state. This happens when
    // `append_command` throws an error, because some changes to the internal state have already
    // been made at that point and can't be reverted.
    // TODO: throw the error in `append_command` _before_ any state changes are made,
    // so that this is no longer needed.
    is_poisoned: bool,

    // True if we're a secondary command buffer.
    is_secondary: bool,
}

// # How pipeline stages work in Vulkan
//
// Imagine you create a command buffer that contains 10 dispatch commands, and submit that command
// buffer. According to the Vulkan specs, the implementation is free to execute the 10 commands
// simultaneously.
//
// Now imagine that the command buffer contains 10 draw commands instead. Contrary to the dispatch
// commands, the draw pipeline contains multiple stages: draw indirect, vertex input, vertex shader,
// ..., fragment shader, late fragment test, color output. When there are multiple stages, the
// implementations must start and end the stages in order. In other words it can start the draw
// indirect stage of all 10 commands, then start the vertex input stage of all 10 commands, and so
// on. But it can't for example start the fragment shader stage of a command before starting the
// vertex shader stage of another command. Same thing for ending the stages in the right order.
//
// Depending on the type of the command, the pipeline stages are different. Compute shaders use the
// compute stage, while transfer commands use the transfer stage. The compute and transfer stages
// aren't ordered.
//
// When you submit multiple command buffers to a queue, the implementation doesn't do anything in
// particular and behaves as if the command buffers were appended to one another. Therefore if you
// submit a command buffer with 10 dispatch commands, followed with another command buffer with 5
// dispatch commands, then the implementation can perform the 15 commands simultaneously.
//
// ## Introducing barriers
//
// In some situations this is not the desired behaviour. If you add a command that writes to a
// buffer followed with another command that reads that buffer, you don't want them to execute
// simultaneously. Instead you want the second one to wait until the first one is finished. This
// is done by adding a pipeline barrier between the two commands.
//
// A pipeline barriers has a source stage and a destination stage (plus various other things).
// A barrier represents a split in the list of commands. When you add it, the stages of the commands
// before the barrier corresponding to the source stage of the barrier, must finish before the
// stages of the commands after the barrier corresponding to the destination stage of the barrier
// can start.
//
// For example if you add a barrier that transitions from the compute stage to the compute stage,
// then the compute stage of all the commands before the barrier must end before the compute stage
// of all the commands after the barrier can start. This is appropriate for the example about
// writing then reading the same buffer.
//
// ## Batching barriers
//
// Since barriers are "expensive" (as the queue must block), vulkano attempts to group as many
// pipeline barriers as possible into one.
//
// Adding a command to a sync command buffer builder does not immediately add it to the underlying
// command buffer builder. Instead the command is added to a queue, and the builder keeps a
// prototype of a barrier that must be added before the commands in the queue are flushed.
//
// Whenever you add a command, the builder will find out whether a barrier is needed before the
// command. If so, it will try to merge this barrier with the prototype and add the command to the
// queue. If not possible, the queue will be entirely flushed and the command added to a fresh new
// queue with a fresh new barrier prototype.

impl fmt::Debug for SyncCommandBufferBuilder {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}

/// Error returned if the builder detects that there's an unsolvable conflict.
#[derive(Debug, Clone)]
pub enum SyncCommandBufferBuilderError {
    /// Unsolvable conflict.
    Conflict {
        command1_name: &'static str,
        command1_param: Cow<'static, str>,
        command1_offset: usize,

        command2_name: &'static str,
        command2_param: Cow<'static, str>,
        command2_offset: usize,
    },

    ExecError(CommandBufferExecError),
}

impl error::Error for SyncCommandBufferBuilderError {}

impl fmt::Display for SyncCommandBufferBuilderError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            SyncCommandBufferBuilderError::Conflict { .. } => write!(fmt, "unsolvable conflict"),
            SyncCommandBufferBuilderError::ExecError(err) => err.fmt(fmt),
        }
    }
}

impl From<CommandBufferExecError> for SyncCommandBufferBuilderError {
    #[inline]
    fn from(val: CommandBufferExecError) -> Self {
        SyncCommandBufferBuilderError::ExecError(val)
    }
}

// List of commands stored inside a `SyncCommandBufferBuilder`.
struct Commands {
    // Only the commands before `first_unflushed` have already been sent to the inner
    // `UnsafeCommandBufferBuilder`.
    first_unflushed: usize,

    // If we're currently inside a render pass, contains the index of the `CmdBeginRenderPass`
    // command.
    latest_render_pass_enter: Option<usize>,

    // The actual list.
    commands: Vec<Box<dyn Command + Send + Sync>>,
}

// Trait for single commands within the list of commands.
pub trait Command {
    // Returns a user-friendly name for the command, for error reporting purposes.
    fn name(&self) -> &'static str;

    // Sends the command to the `UnsafeCommandBufferBuilder`. Calling this method twice on the same
    // object will likely lead to a panic.
    unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder);

    // Turns this command into a `FinalCommand`.
    fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync>;

    // Gives access to the `num`th buffer used by the command.
    fn buffer(&self, _num: usize) -> &dyn BufferAccess {
        panic!()
    }

    // Gives access to the `num`th image used by the command.
    fn image(&self, _num: usize) -> &dyn ImageAccess {
        panic!()
    }

    // Returns a user-friendly name for the `num`th buffer used by the command, for error
    // reporting purposes.
    fn buffer_name(&self, _num: usize) -> Cow<'static, str> {
        panic!()
    }

    // Returns a user-friendly name for the `num`th image used by the command, for error
    // reporting purposes.
    fn image_name(&self, _num: usize) -> Cow<'static, str> {
        panic!()
    }
}

struct CmdPipelineBarrier;

impl Command for CmdPipelineBarrier {
    fn name(&self) -> &'static str {
        "vkCmdPipelineBarrier"
    }

    unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {}

    fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
        struct Fin;
        impl FinalCommand for Fin {
            fn name(&self) -> &'static str {
                "vkCmdPipelineBarrier"
            }
        }
        Box::new(Fin)
    }
}

/// Type of resource whose state is to be tracked.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum KeyTy {
    Buffer,
    Image,
}

// Identifies a resource within the list of commands.
#[derive(Clone, Copy, Debug)]
struct ResourceLocation {
    // Index of the command that holds the resource.
    command_id: usize,
    // Index of the resource within the command.
    resource_index: usize,
}

// Key that identifies a resource. Implements `PartialEq`, `Eq` and `Hash` so that two resources
// that conflict with each other compare equal.
//
// This works by holding an Arc to the list of commands and the index of the command that holds
// the resource.
struct BuilderKey {
    // Same `Arc` as in the `SyncCommandBufferBuilder`.
    commands: Arc<Mutex<Commands>>,
    // Index of the command that holds the resource within `commands`.
    command_ids: RefCell<Vec<usize>>,
    // Type of the resource.
    resource_ty: KeyTy,
    // Index of the resource within the command.
    resource_index: usize,
}

impl BuilderKey {
    // Turns this key used by the builder into a key used by the final command buffer.
    // Called when the command buffer is being built.
    fn into_cb_key(
        self,
        final_commands: Arc<Vec<Box<dyn FinalCommand + Send + Sync>>>,
    ) -> CbKey<'static> {
        CbKey::Command {
            commands: final_commands,
            command_ids: self.command_ids.borrow().clone(),
            resource_ty: self.resource_ty,
            resource_index: self.resource_index,
        }
    }

    #[inline]
    fn conflicts_buffer(&self, commands_lock: &Commands, buf: &dyn BufferAccess) -> bool {
        // TODO: put the conflicts_* methods directly on the Command trait to avoid an indirect call?
        match self.resource_ty {
            KeyTy::Buffer => self.command_ids.borrow().iter().any(|command_id| {
                let c = &commands_lock.commands[*command_id];
                c.buffer(self.resource_index).conflicts_buffer(buf)
            }),
            KeyTy::Image => self.command_ids.borrow().iter().any(|command_id| {
                let c = &commands_lock.commands[*command_id];
                c.image(self.resource_index).conflicts_buffer(buf)
            }),
        }
    }

    #[inline]
    fn conflicts_image(&self, commands_lock: &Commands, img: &dyn ImageAccess) -> bool {
        // TODO: put the conflicts_* methods directly on the Command trait to avoid an indirect call?
        match self.resource_ty {
            KeyTy::Buffer => self.command_ids.borrow().iter().any(|command_id| {
                let c = &commands_lock.commands[*command_id];
                c.buffer(self.resource_index).conflicts_image(img)
            }),
            KeyTy::Image => self.command_ids.borrow().iter().any(|command_id| {
                let c = &commands_lock.commands[*command_id];
                c.image(self.resource_index).conflicts_image(img)
            }),
        }
    }
}

impl PartialEq for BuilderKey {
    #[inline]
    fn eq(&self, other: &BuilderKey) -> bool {
        debug_assert!(Arc::ptr_eq(&self.commands, &other.commands));
        let commands_lock = self.commands.lock().unwrap();

        match other.resource_ty {
            KeyTy::Buffer => other.command_ids.borrow().iter().any(|command_id| {
                let c = &commands_lock.commands[*command_id];
                self.conflicts_buffer(&commands_lock, c.buffer(other.resource_index))
            }),
            KeyTy::Image => other.command_ids.borrow().iter().any(|command_id| {
                let c = &commands_lock.commands[*command_id];
                self.conflicts_image(&commands_lock, c.image(other.resource_index))
            }),
        }
    }
}

impl Eq for BuilderKey {}

impl Hash for BuilderKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let commands_lock = self.commands.lock().unwrap();

        match self.resource_ty {
            KeyTy::Buffer => {
                let c = &commands_lock.commands[self.command_ids.borrow()[0]];
                c.buffer(self.resource_index).conflict_key().hash(state)
            }
            KeyTy::Image => {
                let c = &commands_lock.commands[self.command_ids.borrow()[0]];
                c.image(self.resource_index).conflict_key().hash(state);
                c.image(self.resource_index)
                    .current_miplevels_access()
                    .hash(state);
                c.image(self.resource_index)
                    .current_layer_levels_access()
                    .hash(state);
            }
        }
    }
}

// State of a resource during the building of the command buffer.
#[derive(Debug, Clone)]
struct ResourceState {
    // Memory access of the command that last used this resource.
    memory: PipelineMemoryAccess,

    // True if the resource was used in exclusive mode at any point during the building of the
    // command buffer. Also true if an image layout transition or queue transfer has been performed.
    exclusive_any: bool,

    // Layout at the first use of the resource by the command buffer. Can be `Undefined` if we
    // don't care.
    initial_layout: ImageLayout,

    // Current layout at this stage of the building.
    current_layout: ImageLayout,
}

impl ResourceState {
    // Turns this `ResourceState` into a `ResourceFinalState`. Called when the command buffer is
    // being built.
    #[inline]
    fn finalize(self) -> ResourceFinalState {
        ResourceFinalState {
            final_stages: self.memory.stages,
            final_access: self.memory.access,
            exclusive: self.exclusive_any,
            initial_layout: self.initial_layout,
            final_layout: self.current_layout,
        }
    }
}

impl SyncCommandBufferBuilder {
    /// Builds a new `SyncCommandBufferBuilder`. The parameters are the same as the
    /// `UnsafeCommandBufferBuilder::new` function.
    ///
    /// # Safety
    ///
    /// See `UnsafeCommandBufferBuilder::new()`.
    pub unsafe fn new<R, F>(
        pool_alloc: &UnsafeCommandPoolAlloc,
        kind: Kind<R, F>,
        flags: Flags,
    ) -> Result<SyncCommandBufferBuilder, OomError>
    where
        R: RenderPassAbstract,
        F: FramebufferAbstract,
    {
        let (is_secondary, inside_render_pass) = match kind {
            Kind::Primary => (false, false),
            Kind::Secondary {
                ref render_pass, ..
            } => (true, render_pass.is_some()),
        };

        let cmd = UnsafeCommandBufferBuilder::new(pool_alloc, kind, flags)?;
        Ok(SyncCommandBufferBuilder::from_unsafe_cmd(
            cmd,
            is_secondary,
            inside_render_pass,
        ))
    }

    /// Builds a `SyncCommandBufferBuilder` from an existing `UnsafeCommandBufferBuilder`.
    ///
    /// # Safety
    ///
    /// See `UnsafeCommandBufferBuilder::new()`.
    ///
    /// In addition to this, the `UnsafeCommandBufferBuilder` should be empty. If it isn't, then
    /// you must take into account the fact that the `SyncCommandBufferBuilder` won't be aware of
    /// any existing resource usage.
    #[inline]
    pub unsafe fn from_unsafe_cmd(
        cmd: UnsafeCommandBufferBuilder,
        is_secondary: bool,
        inside_render_pass: bool,
    ) -> SyncCommandBufferBuilder {
        let latest_render_pass_enter = if inside_render_pass { Some(0) } else { None };

        SyncCommandBufferBuilder {
            inner: cmd,
            buffers: Vec::new(),
            images: Vec::new(),
            resources: FnvHashMap::default(),
            pending_barrier: UnsafeCommandBufferBuilderPipelineBarrier::new(),
            commands: Arc::new(Mutex::new(Commands {
                first_unflushed: 0,
                latest_render_pass_enter,
                commands: Vec::new(),
            })),
            barriers: Vec::new(),
            is_poisoned: false,
            is_secondary,
        }
    }

    // Adds a command to be processed by the builder.
    //
    // The `resources` argument should contain each buffer or image used by the command.
    // The function will take care of handling the pipeline barrier or flushing.
    //
    // - The index of the resource within the `resources` slice maps to the resource accessed
    //   through `Command::buffer(..)` or `Command::image(..)`.
    // - `PipelineMemoryAccess` must match the way the resource has been used.
    // - `start_layout` and `end_layout` designate the image layout that the image is expected to be
    //   in when the command starts, and the image layout that the image will be transitioned to
    //   during the command. When it comes to buffers, you should pass `Undefined` for both.
    #[inline]
    pub(super) fn append_command<C>(
        &mut self,
        command: C,
        resources: &[(
            KeyTy,
            Option<(PipelineMemoryAccess, ImageLayout, ImageLayout)>,
        )],
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        C: Command + Send + Sync + 'static,
    {
        // TODO: see comment for the `is_poisoned` member in the struct
        assert!(
            !self.is_poisoned,
            "The builder has been put in an inconsistent state by a previous error"
        );

        // Note that we don't submit the command to the inner command buffer yet.
        let (latest_command_id, end) = {
            let mut commands_lock = self.commands.lock().unwrap();
            commands_lock.commands.push(Box::new(command));
            let latest_command_id = commands_lock.commands.len() - 1;
            let end = commands_lock
                .latest_render_pass_enter
                .unwrap_or(latest_command_id);
            (latest_command_id, end)
        };
        let mut last_cmd_buffer = 0;
        let mut last_cmd_image = 0;

        for &(resource_ty, resource) in resources {
            if let Some((memory, start_layout, end_layout)) = resource {
                // Anti-dumbness checks.
                debug_assert!(memory.exclusive || start_layout == end_layout);
                debug_assert!(memory.access.is_compatible_with(&memory.stages));
                debug_assert!(resource_ty != KeyTy::Image || end_layout != ImageLayout::Undefined);
                debug_assert!(
                    resource_ty != KeyTy::Buffer || start_layout == ImageLayout::Undefined
                );
                debug_assert!(resource_ty != KeyTy::Buffer || end_layout == ImageLayout::Undefined);
                debug_assert_ne!(end_layout, ImageLayout::Preinitialized);

                let resource_index = match resource_ty {
                    KeyTy::Buffer => last_cmd_buffer,
                    KeyTy::Image => last_cmd_image,
                };

                let key = BuilderKey {
                    commands: self.commands.clone(),
                    command_ids: RefCell::new(vec![latest_command_id]),
                    resource_ty,
                    resource_index,
                };

                // Note that the call to `entry()` will lock the mutex, so we can't keep it locked
                // throughout the function.
                match self.resources.entry(key) {
                    // Situation where this resource was used before in this command buffer.
                    Entry::Occupied(entry) => {
                        // `collision_cmd_ids` contains the IDs of the commands that we are potentially
                        // colliding with.
                        let collision_cmd_ids = entry.key().command_ids.borrow().clone();
                        debug_assert!(collision_cmd_ids.iter().all(|id| *id <= latest_command_id));

                        let entry_key_resource_index = entry.key().resource_index;
                        let entry_key_resource_ty = entry.key().resource_ty;

                        // Find out if we have a collision with the pending commands.
                        if memory.exclusive
                            || entry.get().memory.exclusive
                            || entry.get().current_layout != start_layout
                        {
                            // Collision found between `latest_command_id` and `collision_cmd_id`.

                            // We now want to modify the current pipeline barrier in order to handle the
                            // collision. But since the pipeline barrier is going to be submitted before
                            // the flushed commands, it would be a mistake if `collision_cmd_id` hasn't
                            // been flushed yet.
                            let first_unflushed_cmd_id = {
                                let commands_lock = self.commands.lock().unwrap();
                                commands_lock.first_unflushed
                            };

                            if collision_cmd_ids
                                .iter()
                                .any(|command_id| *command_id >= first_unflushed_cmd_id)
                                || entry.get().current_layout != start_layout
                            {
                                unsafe {
                                    // Flush the pending barrier.
                                    self.inner.pipeline_barrier(&self.pending_barrier);
                                    self.pending_barrier =
                                        UnsafeCommandBufferBuilderPipelineBarrier::new();

                                    // Flush the commands if possible, or return an error if not possible.
                                    {
                                        let mut commands_lock = self.commands.lock().unwrap();
                                        let start = commands_lock.first_unflushed;
                                        self.barriers.push(start); // Track inserted barriers

                                        if let Some(collision_cmd_id) = collision_cmd_ids
                                            .iter()
                                            .find(|command_id| **command_id >= end)
                                        {
                                            // TODO: see comment for the `is_poisoned` member in the struct
                                            self.is_poisoned = true;

                                            let cmd1 = &commands_lock.commands[*collision_cmd_id];
                                            let cmd2 = &commands_lock.commands[latest_command_id];

                                            return Err(SyncCommandBufferBuilderError::Conflict {
                                                command1_name: cmd1.name(),
                                                command1_param: match entry_key_resource_ty {
                                                    KeyTy::Buffer => {
                                                        cmd1.buffer_name(entry_key_resource_index)
                                                    }
                                                    KeyTy::Image => {
                                                        cmd1.image_name(entry_key_resource_index)
                                                    }
                                                },
                                                command1_offset: *collision_cmd_id,

                                                command2_name: cmd2.name(),
                                                command2_param: match resource_ty {
                                                    KeyTy::Buffer => {
                                                        cmd2.buffer_name(resource_index)
                                                    }
                                                    KeyTy::Image => cmd2.image_name(resource_index),
                                                },
                                                command2_offset: latest_command_id,
                                            });
                                        }
                                        for command in &mut commands_lock.commands[start..end] {
                                            command.send(&mut self.inner);
                                        }
                                        commands_lock.first_unflushed = end;
                                    }
                                }
                            }

                            entry.key().command_ids.borrow_mut().push(latest_command_id);
                            let entry = entry.into_mut();

                            // Modify the pipeline barrier to handle the collision.
                            unsafe {
                                let commands_lock = self.commands.lock().unwrap();
                                match resource_ty {
                                    KeyTy::Buffer => {
                                        let buf = commands_lock.commands[latest_command_id]
                                            .buffer(resource_index);

                                        let b = &mut self.pending_barrier;
                                        b.add_buffer_memory_barrier(
                                            buf,
                                            entry.memory.stages,
                                            entry.memory.access,
                                            memory.stages,
                                            memory.access,
                                            true,
                                            None,
                                            0,
                                            buf.size(),
                                        );
                                    }

                                    KeyTy::Image => {
                                        let img = commands_lock.commands[latest_command_id]
                                            .image(resource_index);

                                        let b = &mut self.pending_barrier;
                                        b.add_image_memory_barrier(
                                            img,
                                            img.current_miplevels_access(),
                                            img.current_layer_levels_access(),
                                            entry.memory.stages,
                                            entry.memory.access,
                                            memory.stages,
                                            memory.access,
                                            true,
                                            None,
                                            entry.current_layout,
                                            start_layout,
                                        );
                                    }
                                };
                            }

                            // Update state.
                            entry.memory = memory;
                            entry.exclusive_any = true;
                            if memory.exclusive || end_layout != ImageLayout::Undefined {
                                // Only modify the layout in case of a write, because buffer operations
                                // pass `Undefined` for the layout. While a buffer write *must* set the
                                // layout to `Undefined`, a buffer read must not touch it.
                                entry.current_layout = end_layout;
                            }
                        } else {
                            // There is no collision. Simply merge the stages and accesses.
                            // TODO: what about simplifying the newly-constructed stages/accesses?
                            //       this would simplify the job of the driver, but is it worth it?
                            let entry = entry.into_mut();
                            entry.memory.stages |= memory.stages;
                            entry.memory.access |= memory.access;
                        }
                    }

                    // Situation where this is the first time we use this resource in this command buffer.
                    Entry::Vacant(entry) => {
                        // We need to perform some tweaks if the initial layout requirement of the image
                        // is different from the first layout usage.
                        let mut actually_exclusive = memory.exclusive;
                        let mut actual_start_layout = start_layout;

                        if !self.is_secondary
                            && resource_ty == KeyTy::Image
                            && start_layout != ImageLayout::Undefined
                            && start_layout != ImageLayout::Preinitialized
                        {
                            let commands_lock = self.commands.lock().unwrap();
                            let img =
                                commands_lock.commands[latest_command_id].image(resource_index);
                            let initial_layout_requirement = img.initial_layout_requirement();

                            // Checks if the image is initialized and transitions it
                            // if it isn't
                            let is_layout_initialized = img.is_layout_initialized();

                            if initial_layout_requirement != start_layout || !is_layout_initialized
                            {
                                // Note that we transition from `bottom_of_pipe`, which means that we
                                // wait for all the previous commands to be entirely finished. This is
                                // suboptimal, but:
                                //
                                // - If we're at the start of the command buffer we have no choice anyway,
                                //   because we have no knowledge about what comes before.
                                // - If we're in the middle of the command buffer, this pipeline is going
                                //   to be merged with an existing barrier. While it may still be
                                //   suboptimal in some cases, in the general situation it will be ok.
                                //
                                unsafe {
                                    let from_layout = if is_layout_initialized {
                                        actually_exclusive = true;
                                        initial_layout_requirement
                                    } else {
                                        if img.preinitialized_layout() {
                                            ImageLayout::Preinitialized
                                        } else {
                                            ImageLayout::Undefined
                                        }
                                    };
                                    if initial_layout_requirement != start_layout {
                                        actual_start_layout = initial_layout_requirement;
                                    }
                                    let b = &mut self.pending_barrier;
                                    b.add_image_memory_barrier(
                                        img,
                                        img.current_miplevels_access(),
                                        img.current_layer_levels_access(),
                                        PipelineStages {
                                            bottom_of_pipe: true,
                                            ..PipelineStages::none()
                                        },
                                        AccessFlagBits::none(),
                                        memory.stages,
                                        memory.access,
                                        true,
                                        None,
                                        from_layout,
                                        start_layout,
                                    );
                                    img.layout_initialized();
                                }
                            }
                        }

                        entry.insert(ResourceState {
                            memory: PipelineMemoryAccess {
                                stages: memory.stages,
                                access: memory.access,
                                exclusive: actually_exclusive,
                            },
                            exclusive_any: actually_exclusive,
                            initial_layout: actual_start_layout,
                            current_layout: end_layout, // TODO: what if we reach the end with Undefined? that's not correct?
                        });
                    }
                }

                // Add the resources to the lists
                // TODO: Perhaps any barriers for a resource in the secondary command buffer will "protect"
                // its accesses so the primary needs less strict barriers.
                // Less barriers is more efficient, so worth investigating!
                let location = ResourceLocation {
                    command_id: latest_command_id,
                    resource_index,
                };

                match resource_ty {
                    KeyTy::Buffer => {
                        self.buffers.push((location, memory));
                        last_cmd_buffer += 1;
                    }
                    KeyTy::Image => {
                        self.images
                            .push((location, memory, start_layout, end_layout));
                        last_cmd_image += 1;
                    }
                }
            } else {
                match resource_ty {
                    KeyTy::Buffer => {
                        last_cmd_buffer += 1;
                    }
                    KeyTy::Image => {
                        last_cmd_image += 1;
                    }
                }
            }
        }

        Ok(())
    }

    // Call this when the previous command entered a render pass.
    #[inline]
    pub(super) fn prev_cmd_entered_render_pass(&mut self) {
        // TODO: see comment for the `is_poisoned` member in the struct
        assert!(
            !self.is_poisoned,
            "The builder has been put in an inconsistent state by a previous error"
        );

        let mut cmd_lock = self.commands.lock().unwrap();
        cmd_lock.latest_render_pass_enter = Some(cmd_lock.commands.len() - 1);
    }

    // Call this when the previous command left a render pass.
    #[inline]
    pub(super) fn prev_cmd_left_render_pass(&mut self) {
        // TODO: see comment for the `is_poisoned` member in the struct
        assert!(
            !self.is_poisoned,
            "The builder has been put in an inconsistent state by a previous error"
        );

        let mut cmd_lock = self.commands.lock().unwrap();
        debug_assert!(cmd_lock.latest_render_pass_enter.is_some());
        cmd_lock.latest_render_pass_enter = None;
    }

    /// Builds the command buffer and turns it into a `SyncCommandBuffer`.
    #[inline]
    pub fn build(mut self) -> Result<SyncCommandBuffer, OomError> {
        // TODO: see comment for the `is_poisoned` member in the struct
        assert!(
            !self.is_poisoned,
            "The builder has been put in an inconsistent state by a previous error"
        );

        let mut commands_lock = self.commands.lock().unwrap();
        debug_assert!(
            commands_lock.latest_render_pass_enter.is_none() || self.pending_barrier.is_empty()
        );

        // The commands that haven't been sent to the inner command buffer yet need to be sent.
        unsafe {
            self.inner.pipeline_barrier(&self.pending_barrier);
            let start = commands_lock.first_unflushed;
            self.barriers.push(start); // Track inserted barriers
            for command in &mut commands_lock.commands[start..] {
                command.send(&mut self.inner);
            }
        }

        // Transition images to their desired final layout.
        if !self.is_secondary {
            unsafe {
                // TODO: this could be optimized by merging the barrier with the barrier above?
                let mut barrier = UnsafeCommandBufferBuilderPipelineBarrier::new();

                for (key, state) in &mut self.resources {
                    if key.resource_ty != KeyTy::Image {
                        continue;
                    }

                    let img = commands_lock.commands[key.command_ids.borrow()[0]]
                        .image(key.resource_index);
                    let requested_layout = img.final_layout_requirement();
                    if requested_layout == state.current_layout {
                        continue;
                    }

                    barrier.add_image_memory_barrier(
                        img,
                        img.current_miplevels_access(),
                        img.current_layer_levels_access(),
                        state.memory.stages,
                        state.memory.access,
                        PipelineStages {
                            top_of_pipe: true,
                            ..PipelineStages::none()
                        },
                        AccessFlagBits::none(),
                        true,
                        None, // TODO: queue transfers?
                        state.current_layout,
                        requested_layout,
                    );

                    state.exclusive_any = true;
                    state.current_layout = requested_layout;
                }

                self.inner.pipeline_barrier(&barrier);
            }
        }

        // Turns the commands into a list of "final commands" that are slimmer.
        let final_commands = {
            let mut final_commands = Vec::with_capacity(commands_lock.commands.len());
            for command in commands_lock.commands.drain(..) {
                final_commands.push(command.into_final_command());
            }
            Arc::new(final_commands)
        };

        // Build the final resources states.
        let final_resources_states: FnvHashMap<_, _> = {
            self.resources
                .into_iter()
                .map(|(resource, state)| {
                    (
                        resource.into_cb_key(final_commands.clone()),
                        state.finalize(),
                    )
                })
                .collect()
        };

        Ok(SyncCommandBuffer {
            inner: self.inner.build()?,
            buffers: self.buffers,
            images: self.images,
            resources: final_resources_states,
            commands: final_commands,
            barriers: self.barriers,
        })
    }
}

unsafe impl DeviceOwned for SyncCommandBufferBuilder {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

/// Command buffer built from a `SyncCommandBufferBuilder` that provides utilities to handle
/// synchronization.
pub struct SyncCommandBuffer {
    // The actual Vulkan command buffer.
    inner: UnsafeCommandBuffer,

    // Resources and their accesses. Used for executing secondary command buffers in a primary.
    buffers: Vec<(ResourceLocation, PipelineMemoryAccess)>,
    images: Vec<(
        ResourceLocation,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
    )>,

    // State of all the resources used by this command buffer.
    resources: FnvHashMap<CbKey<'static>, ResourceFinalState>,

    // List of commands used by the command buffer. Used to hold the various resources that are
    // being used. Each element of `resources` has a copy of this `Arc`, but we need to keep one
    // here in case `resources` is empty.
    commands: Arc<Vec<Box<dyn FinalCommand + Send + Sync>>>,

    // Locations within commands that pipeline barriers were inserted. For debugging purposes.
    // TODO: present only in cfg(debug_assertions)?
    barriers: Vec<usize>,
}

impl SyncCommandBuffer {
    /// Tries to lock the resources used by the command buffer.
    ///
    /// > **Note**: You should call this in the implementation of the `CommandBuffer` trait.
    pub fn lock_submit(
        &self,
        future: &dyn GpuFuture,
        queue: &Queue,
    ) -> Result<(), CommandBufferExecError> {
        // Number of resources in `self.resources` that have been successfully locked.
        let mut locked_resources = 0;
        // Final return value of this function.
        let mut ret_value = Ok(());

        // Try locking resources. Updates `locked_resources` and `ret_value`, and break if an error
        // happens.
        for (key, entry) in self.resources.iter() {
            let (command_ids, resource_ty, resource_index) = match *key {
                CbKey::Command {
                    ref command_ids,
                    resource_ty,
                    resource_index,
                    ..
                } => (command_ids, resource_ty, resource_index),
                _ => unreachable!(),
            };

            let command = &self.commands[command_ids[0]];

            match resource_ty {
                KeyTy::Buffer => {
                    let buf = command.buffer(resource_index);

                    // Because try_gpu_lock needs to be called first,
                    // this should never return Ok without first returning Err
                    let prev_err = match future.check_buffer_access(&buf, entry.exclusive, queue) {
                        Ok(_) => {
                            unsafe {
                                buf.increase_gpu_lock();
                            }
                            locked_resources += 1;
                            continue;
                        }
                        Err(err) => err,
                    };

                    match (buf.try_gpu_lock(entry.exclusive, queue), prev_err) {
                        (Ok(_), _) => (),
                        (Err(err), AccessCheckError::Unknown)
                        | (_, AccessCheckError::Denied(err)) => {
                            ret_value = Err(CommandBufferExecError::AccessError {
                                error: err,
                                command_name: command.name().into(),
                                command_param: command.buffer_name(resource_index),
                                command_offset: command_ids[0],
                            });
                            break;
                        }
                    };
                }

                KeyTy::Image => {
                    let img = command.image(resource_index);

                    let prev_err = match future.check_image_access(
                        img,
                        entry.initial_layout,
                        entry.exclusive,
                        queue,
                    ) {
                        Ok(_) => {
                            unsafe {
                                img.increase_gpu_lock();
                            }
                            locked_resources += 1;
                            continue;
                        }
                        Err(err) => err,
                    };

                    match (
                        img.try_gpu_lock(entry.exclusive, entry.initial_layout),
                        prev_err,
                    ) {
                        (Ok(_), _) => (),
                        (Err(err), AccessCheckError::Unknown)
                        | (_, AccessCheckError::Denied(err)) => {
                            ret_value = Err(CommandBufferExecError::AccessError {
                                error: err,
                                command_name: command.name().into(),
                                command_param: command.image_name(resource_index),
                                command_offset: command_ids[0],
                            });
                            break;
                        }
                    };
                }
            }

            locked_resources += 1;
        }

        // If we are going to return an error, we have to unlock all the resources we locked above.
        if let Err(_) = ret_value {
            for (key, val) in self.resources.iter().take(locked_resources) {
                let (command_ids, resource_ty, resource_index) = match *key {
                    CbKey::Command {
                        ref command_ids,
                        resource_ty,
                        resource_index,
                        ..
                    } => (command_ids, resource_ty, resource_index),
                    _ => unreachable!(),
                };

                let command = &self.commands[command_ids[0]];

                match resource_ty {
                    KeyTy::Buffer => {
                        let buf = command.buffer(resource_index);
                        unsafe {
                            buf.unlock();
                        }
                    }

                    KeyTy::Image => {
                        let command = &self.commands[command_ids[0]];
                        let img = command.image(resource_index);
                        let trans = if val.final_layout != val.initial_layout {
                            Some(val.final_layout)
                        } else {
                            None
                        };
                        unsafe {
                            img.unlock(trans);
                        }
                    }
                }
            }
        }

        // TODO: pipeline barriers if necessary?

        ret_value
    }

    /// Unlocks the resources used by the command buffer.
    ///
    /// > **Note**: You should call this in the implementation of the `CommandBuffer` trait.
    ///
    /// # Safety
    ///
    /// The command buffer must have been successfully locked with `lock_submit()`.
    ///
    pub unsafe fn unlock(&self) {
        for (key, val) in self.resources.iter() {
            let (command_ids, resource_ty, resource_index) = match *key {
                CbKey::Command {
                    ref command_ids,
                    resource_ty,
                    resource_index,
                    ..
                } => (command_ids, resource_ty, resource_index),
                _ => unreachable!(),
            };

            let command = &self.commands[command_ids[0]];

            match resource_ty {
                KeyTy::Buffer => {
                    let buf = command.buffer(resource_index);
                    buf.unlock();
                }

                KeyTy::Image => {
                    let img = command.image(resource_index);
                    let trans = if val.final_layout != val.initial_layout {
                        Some(val.final_layout)
                    } else {
                        None
                    };
                    img.unlock(trans);
                }
            }
        }
    }

    /// Checks whether this command buffer has access to a buffer.
    ///
    /// > **Note**: Suitable when implementing the `CommandBuffer` trait.
    #[inline]
    pub fn check_buffer_access(
        &self,
        buffer: &dyn BufferAccess,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        // TODO: check the queue family

        if let Some(value) = self.resources.get(&CbKey::BufferRef(buffer)) {
            if !value.exclusive && exclusive {
                return Err(AccessCheckError::Unknown);
            }

            return Ok(Some((value.final_stages, value.final_access)));
        }

        Err(AccessCheckError::Unknown)
    }

    /// Checks whether this command buffer has access to an image.
    ///
    /// > **Note**: Suitable when implementing the `CommandBuffer` trait.
    #[inline]
    pub fn check_image_access(
        &self,
        image: &dyn ImageAccess,
        layout: ImageLayout,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        // TODO: check the queue family

        if let Some(value) = self.resources.get(&CbKey::ImageRef(image)) {
            if layout != ImageLayout::Undefined && value.final_layout != layout {
                return Err(AccessCheckError::Denied(
                    AccessError::UnexpectedImageLayout {
                        allowed: value.final_layout,
                        requested: layout,
                    },
                ));
            }

            if !value.exclusive && exclusive {
                return Err(AccessCheckError::Unknown);
            }

            return Ok(Some((value.final_stages, value.final_access)));
        }

        Err(AccessCheckError::Unknown)
    }

    #[inline]
    pub fn num_buffers(&self) -> usize {
        self.buffers.len()
    }

    #[inline]
    pub fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, PipelineMemoryAccess)> {
        self.buffers.get(index).map(|(location, memory)| {
            let cmd = &self.commands[location.command_id];
            (cmd.buffer(location.resource_index), *memory)
        })
    }

    #[inline]
    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    #[inline]
    pub fn image(
        &self,
        index: usize,
    ) -> Option<(
        &dyn ImageAccess,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
    )> {
        self.images
            .get(index)
            .map(|(location, memory, start_layout, end_layout)| {
                let cmd = &self.commands[location.command_id];
                (
                    cmd.image(location.resource_index),
                    *memory,
                    *start_layout,
                    *end_layout,
                )
            })
    }
}

impl AsRef<UnsafeCommandBuffer> for SyncCommandBuffer {
    #[inline]
    fn as_ref(&self) -> &UnsafeCommandBuffer {
        &self.inner
    }
}

unsafe impl DeviceOwned for SyncCommandBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

// Usage of a resource in a finished command buffer.
#[derive(Debug, Clone)]
struct ResourceFinalState {
    // Stages of the last command that uses the resource.
    final_stages: PipelineStages,
    // Access for the last command that uses the resource.
    final_access: AccessFlagBits,

    // True if the resource is used in exclusive mode.
    exclusive: bool,

    // Layout that an image must be in at the start of the command buffer. Can be `Undefined` if we
    // don't care.
    initial_layout: ImageLayout,

    // Layout the image will be in at the end of the command buffer.
    final_layout: ImageLayout, // TODO: maybe wrap in an Option to mean that the layout doesn't change? because of buffers?
}

/// Equivalent to `Command`, but with less methods. Typically contains less things than the
/// `Command` it comes from.
pub trait FinalCommand {
    // Returns a user-friendly name for the command, for error reporting purposes.
    fn name(&self) -> &'static str;

    // Gives access to the `num`th buffer used by the command.
    fn buffer(&self, _num: usize) -> &dyn BufferAccess {
        panic!()
    }

    // Gives access to the `num`th image used by the command.
    fn image(&self, _num: usize) -> &dyn ImageAccess {
        panic!()
    }

    // Returns a user-friendly name for the `num`th buffer used by the command, for error
    // reporting purposes.
    fn buffer_name(&self, _num: usize) -> Cow<'static, str> {
        panic!()
    }

    // Returns a user-friendly name for the `num`th image used by the command, for error
    // reporting purposes.
    fn image_name(&self, _num: usize) -> Cow<'static, str> {
        panic!()
    }
}

impl FinalCommand for &'static str {
    fn name(&self) -> &'static str {
        *self
    }
}

// Equivalent of `BuilderKey` for a finished command buffer.
//
// In addition to this, it also add two other variants which are `BufferRef` and `ImageRef`. These
// variants are used in order to make it possible to compare a `CbKey` stored in the
// `SyncCommandBuffer` with a temporarily-created `CbKey`. The Rust HashMap doesn't allow us to do
// that otherwise.
//
// You should never store a `BufferRef` or a `ImageRef` inside the `SyncCommandBuffer`.
enum CbKey<'a> {
    // The resource is held in the list of commands.
    Command {
        // Same `Arc` as in the `SyncCommandBufferBuilder`.
        commands: Arc<Vec<Box<dyn FinalCommand + Send + Sync>>>,
        // Index of the command that holds the resource within `commands`.
        command_ids: Vec<usize>,
        // Type of the resource.
        resource_ty: KeyTy,
        // Index of the resource within the command.
        resource_index: usize,
    },

    // Temporary key that holds a reference to a buffer. Should never be stored in the list of
    // resources of `SyncCommandBuffer`.
    BufferRef(&'a dyn BufferAccess),

    // Temporary key that holds a reference to an image. Should never be stored in the list of
    // resources of `SyncCommandBuffer`.
    ImageRef(&'a dyn ImageAccess),
}

// The `CbKey::Command` variants implements `Send` and `Sync`, but the other two variants don't
// because it would be too constraining.
//
// Since only `CbKey::Command` must be stored in the resources hashmap, we force-implement `Send`
// and `Sync` so that the hashmap itself implements `Send` and `Sync`.
unsafe impl<'a> Send for CbKey<'a> {}
unsafe impl<'a> Sync for CbKey<'a> {}

impl<'a> CbKey<'a> {
    #[inline]
    fn conflicts_buffer(
        &self,
        commands_external: Option<&Vec<Box<dyn FinalCommand + Send + Sync>>>,
        buf: &dyn BufferAccess,
    ) -> bool {
        match *self {
            CbKey::Command {
                ref commands,
                ref command_ids,
                resource_ty,
                resource_index,
            } => {
                let commands = commands_external.unwrap_or(commands);

                // TODO: put the conflicts_* methods directly on the FinalCommand trait to avoid an indirect call?
                match resource_ty {
                    KeyTy::Buffer => command_ids.iter().any(|command_id| {
                        let c = &commands[*command_id];
                        c.buffer(resource_index).conflicts_buffer(buf)
                    }),
                    KeyTy::Image => command_ids.iter().any(|command_id| {
                        let c = &commands[*command_id];
                        c.image(resource_index).conflicts_buffer(buf)
                    }),
                }
            }

            CbKey::BufferRef(b) => b.conflicts_buffer(buf),
            CbKey::ImageRef(i) => i.conflicts_buffer(buf),
        }
    }

    #[inline]
    fn conflicts_image(
        &self,
        commands_external: Option<&Vec<Box<dyn FinalCommand + Send + Sync>>>,
        img: &dyn ImageAccess,
    ) -> bool {
        match *self {
            CbKey::Command {
                ref commands,
                ref command_ids,
                resource_ty,
                resource_index,
            } => {
                let commands = commands_external.unwrap_or(commands);

                // TODO: put the conflicts_* methods directly on the Command trait to avoid an indirect call?
                match resource_ty {
                    KeyTy::Buffer => command_ids.iter().any(|command_id| {
                        let c = &commands[*command_id];
                        c.buffer(resource_index).conflicts_image(img)
                    }),
                    KeyTy::Image => command_ids.iter().any(|command_id| {
                        let c = &commands[*command_id];
                        c.image(resource_index).conflicts_image(img)
                    }),
                }
            }

            CbKey::BufferRef(b) => b.conflicts_image(img),
            CbKey::ImageRef(i) => i.conflicts_image(img),
        }
    }
}

impl<'a> PartialEq for CbKey<'a> {
    #[inline]
    fn eq(&self, other: &CbKey) -> bool {
        match *self {
            CbKey::BufferRef(a) => other.conflicts_buffer(None, a),
            CbKey::ImageRef(a) => other.conflicts_image(None, a),
            CbKey::Command {
                ref commands,
                ref command_ids,
                resource_ty,
                resource_index,
            } => {
                let commands_lock = commands; //.lock().unwrap();

                match resource_ty {
                    KeyTy::Buffer => command_ids.iter().any(|command_id| {
                        let c = &commands_lock[*command_id];
                        other.conflicts_buffer(Some(&commands_lock), c.buffer(resource_index))
                    }),
                    KeyTy::Image => command_ids.iter().any(|command_id| {
                        let c = &commands_lock[*command_id];
                        other.conflicts_image(Some(&commands_lock), c.image(resource_index))
                    }),
                }
            }
        }
    }
}

impl<'a> Eq for CbKey<'a> {}

impl<'a> Hash for CbKey<'a> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            CbKey::Command {
                ref commands,
                ref command_ids,
                resource_ty,
                resource_index,
            } => match resource_ty {
                KeyTy::Buffer => {
                    let c = &commands[command_ids[0]];
                    c.buffer(resource_index).conflict_key().hash(state)
                }
                KeyTy::Image => {
                    let c = &commands[command_ids[0]];
                    c.image(resource_index).conflict_key().hash(state)
                }
            },

            CbKey::BufferRef(buf) => buf.conflict_key().hash(state),
            CbKey::ImageRef(img) => img.conflict_key().hash(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SyncCommandBufferBuilder;
    use super::SyncCommandBufferBuilderError;
    use crate::buffer::BufferUsage;
    use crate::buffer::CpuAccessibleBuffer;
    use crate::buffer::ImmutableBuffer;
    use crate::command_buffer::pool::CommandPool;
    use crate::command_buffer::pool::CommandPoolBuilderAlloc;
    use crate::command_buffer::sys::Flags;
    use crate::command_buffer::AutoCommandBufferBuilder;
    use crate::command_buffer::Kind;
    use crate::device::Device;
    use crate::sync::GpuFuture;
    use std::sync::Arc;

    #[test]
    fn basic_creation() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();
            let pool = Device::standard_command_pool(&device, queue.family());
            let pool_builder_alloc = pool.alloc(false, 1).unwrap().next().unwrap();

            assert!(matches!(
                SyncCommandBufferBuilder::new(
                    &pool_builder_alloc.inner(),
                    Kind::primary(),
                    Flags::None,
                ),
                Ok(_)
            ));
        }
    }

    #[test]
    fn basic_conflict() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let pool = Device::standard_command_pool(&device, queue.family());
            let pool_builder_alloc = pool.alloc(false, 1).unwrap().next().unwrap();
            let mut sync = SyncCommandBufferBuilder::new(
                &pool_builder_alloc.inner(),
                Kind::primary(),
                Flags::None,
            )
            .unwrap();
            let buf =
                CpuAccessibleBuffer::from_data(device, BufferUsage::all(), false, 0u32).unwrap();

            assert!(matches!(
                sync.copy_buffer(buf.clone(), buf.clone(), std::iter::once((0, 0, 4))),
                Err(SyncCommandBufferBuilderError::Conflict { .. })
            ));
        }
    }

    #[test]
    fn secondary_conflicting_writes() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            // Create a tiny test buffer
            let (buf, future) = ImmutableBuffer::from_data(
                0u32,
                BufferUsage::transfer_destination(),
                queue.clone(),
            )
            .unwrap();
            future
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

            // Two secondary command buffers that both write to the buffer
            let secondary = (0..2)
                .map(|_| {
                    let mut builder = AutoCommandBufferBuilder::secondary_compute_simultaneous_use(
                        device.clone(),
                        queue.family(),
                    )
                    .unwrap();
                    builder.fill_buffer(buf.clone(), 42u32).unwrap();
                    Arc::new(builder.build().unwrap())
                })
                .collect::<Vec<_>>();

            let pool = Device::standard_command_pool(&device, queue.family());
            let allocs = pool.alloc(false, 2).unwrap().collect::<Vec<_>>();

            {
                let mut builder = SyncCommandBufferBuilder::new(
                    allocs[0].inner(),
                    Kind::primary(),
                    Flags::SimultaneousUse,
                )
                .unwrap();

                // Add both secondary command buffers using separate execute_commands calls.
                secondary.iter().cloned().for_each(|secondary| {
                    let mut ec = builder.execute_commands();
                    ec.add(secondary);
                    ec.submit().unwrap();
                });

                let primary = builder.build().unwrap();
                let names = primary
                    .commands
                    .iter()
                    .map(|c| c.name())
                    .collect::<Vec<_>>();

                // Ensure that the builder added a barrier between the two writes
                assert_eq!(&names, &["vkCmdExecuteCommands", "vkCmdExecuteCommands"]);
                assert_eq!(&primary.barriers, &[0, 1]);
            }

            {
                let mut builder = SyncCommandBufferBuilder::new(
                    allocs[1].inner(),
                    Kind::primary(),
                    Flags::SimultaneousUse,
                )
                .unwrap();

                // Add a single execute_commands for all secondary command buffers at once
                let mut ec = builder.execute_commands();
                secondary.into_iter().for_each(|secondary| {
                    ec.add(secondary);
                });

                // The two writes can't be split up by a barrier because they are part of the same
                // command. Therefore an error.
                // TODO: Would be nice if SyncCommandBufferBuilder would split the commands
                // automatically in order to insert a barrier.
                assert!(matches!(
                    ec.submit(),
                    Err(SyncCommandBufferBuilderError::Conflict { .. })
                ));
            }
        }
    }
}
