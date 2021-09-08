// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub use self::commands::SyncCommandBufferBuilderBindDescriptorSets;
pub use self::commands::SyncCommandBufferBuilderBindVertexBuffer;
pub use self::commands::SyncCommandBufferBuilderExecuteCommands;
use super::Command;
use super::ResourceFinalState;
use super::ResourceKey;
use super::ResourceLocation;
use super::SyncCommandBuffer;
use crate::command_buffer::pool::UnsafeCommandPoolAlloc;
use crate::command_buffer::sys::UnsafeCommandBufferBuilder;
use crate::command_buffer::sys::UnsafeCommandBufferBuilderPipelineBarrier;
use crate::command_buffer::CommandBufferExecError;
use crate::command_buffer::CommandBufferLevel;
use crate::command_buffer::CommandBufferUsage;
use crate::command_buffer::ImageUninitializedSafe;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::image::ImageLayout;
use crate::render_pass::FramebufferAbstract;
use crate::sync::AccessFlags;
use crate::sync::PipelineMemoryAccess;
use crate::sync::PipelineStages;
use crate::OomError;
use commands::CurrentState;
pub use commands::StencilState;
use fnv::FnvHashMap;
use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::error;
use std::fmt;
use std::sync::Arc;

#[path = "commands.rs"]
mod commands;

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

    // Stores all the commands that were added to the sync builder. Some of them are maybe not
    // submitted to the inner builder yet.
    // Each command owns the resources it uses (buffers, images, pipelines, descriptor sets etc.),
    // references to any of these must be indirect in the form of a command index + resource id.
    commands: Vec<Arc<dyn Command + Send + Sync>>,

    // Prototype for the pipeline barrier that must be submitted before flushing the commands
    // in `commands`.
    pending_barrier: UnsafeCommandBufferBuilderPipelineBarrier,

    // Locations within commands that pipeline barriers were inserted. For debugging purposes.
    // TODO: present only in cfg(debug_assertions)?
    barriers: Vec<usize>,

    // Only the commands before `first_unflushed` have already been sent to the inner
    // `UnsafeCommandBufferBuilder`.
    first_unflushed: usize,

    // If we're currently inside a render pass, contains the index of the `CmdBeginRenderPass`
    // command.
    latest_render_pass_enter: Option<usize>,

    // Stores the current state of buffers and images that are in use by the command buffer.
    resources: FnvHashMap<ResourceKey, ResourceState>,

    // Resources and their accesses. Used for executing secondary command buffers in a primary.
    buffers: Vec<(ResourceLocation, PipelineMemoryAccess)>,
    images: Vec<(
        ResourceLocation,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
        ImageUninitializedSafe,
    )>,

    // Current binding/setting state.
    current_state: CurrentState,

    // `true` if the builder has been put in an inconsistent state. This happens when
    // `append_command` throws an error, because some changes to the internal state have already
    // been made at that point and can't be reverted.
    // TODO: throw the error in `append_command` _before_ any state changes are made,
    // so that this is no longer needed.
    is_poisoned: bool,

    // True if we're a secondary command buffer.
    is_secondary: bool,
}

impl SyncCommandBufferBuilder {
    /// Builds a new `SyncCommandBufferBuilder`. The parameters are the same as the
    /// `UnsafeCommandBufferBuilder::new` function.
    ///
    /// # Safety
    ///
    /// See `UnsafeCommandBufferBuilder::new()`.
    pub unsafe fn new<F>(
        pool_alloc: &UnsafeCommandPoolAlloc,
        level: CommandBufferLevel<F>,
        usage: CommandBufferUsage,
    ) -> Result<SyncCommandBufferBuilder, OomError>
    where
        F: FramebufferAbstract,
    {
        let (is_secondary, inside_render_pass) = match level {
            CommandBufferLevel::Primary => (false, false),
            CommandBufferLevel::Secondary(ref inheritance) => {
                (true, inheritance.render_pass.is_some())
            }
        };

        let cmd = UnsafeCommandBufferBuilder::new(pool_alloc, level, usage)?;
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
            commands: Vec::new(),
            pending_barrier: UnsafeCommandBufferBuilderPipelineBarrier::new(),
            barriers: Vec::new(),
            first_unflushed: 0,
            latest_render_pass_enter,
            resources: FnvHashMap::default(),
            buffers: Vec::new(),
            images: Vec::new(),
            current_state: Default::default(),
            is_poisoned: false,
            is_secondary,
        }
    }

    /// Resets the binding/setting state.
    ///
    /// This must be called after any command that changes the state in an undefined way, e.g.
    /// executing a secondary command buffer.
    #[inline]
    pub fn reset_state(&mut self) {
        self.current_state = Default::default();
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
    fn append_command<C>(
        &mut self,
        command: C,
        resources: &[(
            KeyTy,
            Option<(
                PipelineMemoryAccess,
                ImageLayout,
                ImageLayout,
                ImageUninitializedSafe,
            )>,
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
            self.commands.push(Arc::new(command));
            let latest_command_id = self.commands.len() - 1;
            let end = self.latest_render_pass_enter.unwrap_or(latest_command_id);
            (latest_command_id, end)
        };
        let mut last_cmd_buffer = 0;
        let mut last_cmd_image = 0;

        for &(resource_ty, resource) in resources {
            if let Some((memory, start_layout, end_layout, image_uninitialized_safe)) = resource {
                // Anti-dumbness checks.
                debug_assert!(memory.exclusive || start_layout == end_layout);
                debug_assert!(memory.access.is_compatible_with(&memory.stages));
                debug_assert!(resource_ty != KeyTy::Image || end_layout != ImageLayout::Undefined);
                debug_assert!(
                    resource_ty != KeyTy::Buffer || start_layout == ImageLayout::Undefined
                );
                debug_assert!(resource_ty != KeyTy::Buffer || end_layout == ImageLayout::Undefined);
                debug_assert_ne!(end_layout, ImageLayout::Preinitialized);

                let (resource_key, resource_index) = match resource_ty {
                    KeyTy::Buffer => {
                        let buffer = self.commands[latest_command_id].buffer(last_cmd_buffer);
                        (ResourceKey::from(buffer), last_cmd_buffer)
                    }
                    KeyTy::Image => {
                        let image = self.commands[latest_command_id].image(last_cmd_image);
                        (ResourceKey::from(image), last_cmd_image)
                    }
                };

                match self.resources.entry(resource_key) {
                    // Situation where this resource was used before in this command buffer.
                    Entry::Occupied(mut entry) => {
                        // `collision_cmd_ids` contains the IDs of the commands that we are potentially
                        // colliding with.
                        let collision_cmd_ids = &entry.get().command_ids;
                        debug_assert!(collision_cmd_ids.iter().all(|id| *id <= latest_command_id));

                        let entry_key_resource_index = entry.get().resource_index;

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
                            let first_unflushed_cmd_id = self.first_unflushed;

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
                                        let start = self.first_unflushed;
                                        self.barriers.push(start); // Track inserted barriers

                                        if let Some(collision_cmd_id) = collision_cmd_ids
                                            .iter()
                                            .find(|command_id| **command_id >= end)
                                        {
                                            // TODO: see comment for the `is_poisoned` member in the struct
                                            self.is_poisoned = true;

                                            let cmd1 = &self.commands[*collision_cmd_id];
                                            let cmd2 = &self.commands[latest_command_id];

                                            return Err(SyncCommandBufferBuilderError::Conflict {
                                                command1_name: cmd1.name(),
                                                command1_param: match resource_ty {
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
                                        for command in &mut self.commands[start..end] {
                                            command.send(&mut self.inner);
                                        }
                                        self.first_unflushed = end;
                                    }
                                }
                            }

                            entry.get_mut().command_ids.push(latest_command_id);
                            let entry = entry.into_mut();

                            // Modify the pipeline barrier to handle the collision.
                            unsafe {
                                match resource_ty {
                                    KeyTy::Buffer => {
                                        let buf =
                                            self.commands[latest_command_id].buffer(resource_index);

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
                                        let img =
                                            self.commands[latest_command_id].image(resource_index);

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
                            let img = self.commands[latest_command_id].image(resource_index);
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
                                        AccessFlags::none(),
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
                            command_ids: vec![latest_command_id],
                            resource_index,

                            memory: PipelineMemoryAccess {
                                stages: memory.stages,
                                access: memory.access,
                                exclusive: actually_exclusive,
                            },
                            exclusive_any: actually_exclusive,
                            initial_layout: actual_start_layout,
                            current_layout: end_layout, // TODO: what if we reach the end with Undefined? that's not correct?
                            image_uninitialized_safe,
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
                        self.images.push((
                            location,
                            memory,
                            start_layout,
                            end_layout,
                            image_uninitialized_safe,
                        ));
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

    /// Builds the command buffer and turns it into a `SyncCommandBuffer`.
    #[inline]
    pub fn build(mut self) -> Result<SyncCommandBuffer, OomError> {
        // TODO: see comment for the `is_poisoned` member in the struct
        assert!(
            !self.is_poisoned,
            "The builder has been put in an inconsistent state by a previous error"
        );

        debug_assert!(self.latest_render_pass_enter.is_none() || self.pending_barrier.is_empty());

        // The commands that haven't been sent to the inner command buffer yet need to be sent.
        unsafe {
            self.inner.pipeline_barrier(&self.pending_barrier);
            let start = self.first_unflushed;
            self.barriers.push(start); // Track inserted barriers
            for command in &mut self.commands[start..] {
                command.send(&mut self.inner);
            }
        }

        // Transition images to their desired final layout.
        if !self.is_secondary {
            unsafe {
                // TODO: this could be optimized by merging the barrier with the barrier above?
                let mut barrier = UnsafeCommandBufferBuilderPipelineBarrier::new();

                for (key, state) in self
                    .resources
                    .iter_mut()
                    .filter(|(key, _)| matches!(key, ResourceKey::Image(..)))
                {
                    let img = self.commands[state.command_ids[0]].image(state.resource_index);
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
                        AccessFlags::none(),
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

        // Build the final resources states.
        let final_resources_states: FnvHashMap<_, _> = {
            self.resources
                .into_iter()
                .map(|(resource, state)| {
                    let final_state = ResourceFinalState {
                        command_ids: state.command_ids,
                        resource_index: state.resource_index,
                        final_stages: state.memory.stages,
                        final_access: state.memory.access,
                        exclusive: state.exclusive_any,
                        initial_layout: state.initial_layout,
                        final_layout: state.current_layout,
                        image_uninitialized_safe: state.image_uninitialized_safe,
                    };
                    (resource, final_state)
                })
                .collect()
        };

        Ok(SyncCommandBuffer {
            inner: self.inner.build()?,
            buffers: self.buffers,
            images: self.images,
            resources: final_resources_states,
            commands: self.commands,
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

/// Type of resource whose state is to be tracked.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum KeyTy {
    Buffer,
    Image,
}

// State of a resource during the building of the command buffer.
#[derive(Debug, Clone)]
struct ResourceState {
    // Indices of the commands that contain the resource.
    command_ids: Vec<usize>,

    // Index of the resource within the first command in `command_ids`.
    resource_index: usize,

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

    // Extra context of how the image will be used
    image_uninitialized_safe: ImageUninitializedSafe,
}
