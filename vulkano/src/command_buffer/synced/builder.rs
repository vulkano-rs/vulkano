// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{Command, KeyTy, SyncCommandBuffer};
pub use crate::command_buffer::commands::{
    bind_push::{
        SyncCommandBufferBuilderBindDescriptorSets, SyncCommandBufferBuilderBindVertexBuffer,
    },
    secondary::SyncCommandBufferBuilderExecuteCommands,
};
use crate::{
    buffer::{sys::UnsafeBuffer, BufferAccess},
    command_buffer::{
        pool::UnsafeCommandPoolAlloc,
        synced::{BufferFinalState, BufferUse, ImageFinalState, ImageUse},
        sys::{CommandBufferBeginInfo, UnsafeCommandBufferBuilder},
        CommandBufferExecError, CommandBufferLevel,
    },
    descriptor_set::{DescriptorSetResources, DescriptorSetWithOffsets},
    device::{Device, DeviceOwned},
    image::{sys::UnsafeImage, ImageAccess, ImageLayout},
    pipeline::{
        graphics::{
            color_blend::LogicOp,
            depth_stencil::{CompareOp, StencilOps},
            input_assembly::{IndexType, PrimitiveTopology},
            rasterization::{CullMode, DepthBias, FrontFace, LineStipple},
            viewport::{Scissor, Viewport},
        },
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    },
    range_set::RangeSet,
    sync::{
        AccessFlags, BufferMemoryBarrier, DependencyInfo, ImageMemoryBarrier, PipelineMemoryAccess,
        PipelineStages,
    },
    DeviceSize, OomError, VulkanObject,
};
use rangemap::RangeMap;
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    collections::{hash_map::Entry, HashMap},
    error, fmt,
    mem::replace,
    sync::Arc,
};

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
    pub(in crate::command_buffer) commands: Vec<Box<dyn Command>>,

    // Prototype for the pipeline barrier that must be submitted before flushing the commands
    // in `commands`.
    pending_barrier: DependencyInfo,

    // Locations within commands that pipeline barriers were inserted. For debugging purposes.
    // TODO: present only in cfg(debug_assertions)?
    barriers: Vec<usize>,

    // Only the commands before `first_unflushed` have already been sent to the inner
    // `UnsafeCommandBufferBuilder`.
    first_unflushed: usize,

    // If we're currently inside a render pass, contains the index of the `CmdBeginRenderPass`
    // command.
    pub(in crate::command_buffer) latest_render_pass_enter: Option<usize>,

    // Stores the current state of buffers and images that are in use by the command buffer.
    buffers2: HashMap<Arc<UnsafeBuffer>, RangeMap<DeviceSize, Option<BufferState>>>,
    images2: HashMap<Arc<UnsafeImage>, RangeMap<DeviceSize, Option<ImageState>>>,

    // Resources and their accesses. Used for executing secondary command buffers in a primary.
    buffers: Vec<(Arc<dyn BufferAccess>, PipelineMemoryAccess)>,
    images: Vec<(
        Arc<dyn ImageAccess>,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
    )>,

    // Current binding/setting state.
    pub(in crate::command_buffer) current_state: CurrentState,

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
    pub unsafe fn new(
        pool_alloc: &UnsafeCommandPoolAlloc,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<SyncCommandBufferBuilder, OomError> {
        let is_secondary = pool_alloc.level() == CommandBufferLevel::Secondary;
        let inside_render_pass = is_secondary
            && begin_info
                .inheritance_info
                .as_ref()
                .unwrap()
                .render_pass
                .is_some();

        let cmd = UnsafeCommandBufferBuilder::new(pool_alloc, begin_info)?;
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
            pending_barrier: DependencyInfo::default(),
            barriers: Vec::new(),
            first_unflushed: 0,
            latest_render_pass_enter,
            buffers2: HashMap::default(),
            images2: HashMap::default(),
            buffers: Vec::new(),
            images: Vec::new(),
            current_state: Default::default(),
            is_poisoned: false,
            is_secondary,
        }
    }

    /// Returns the binding/setting state.
    #[inline]
    pub fn state(&self) -> CommandBufferState {
        CommandBufferState {
            current_state: &self.current_state,
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
    pub(in crate::command_buffer) fn append_command<C>(
        &mut self,
        command: C,
        resources: impl IntoIterator<
            Item = (
                KeyTy,
                Cow<'static, str>,
                Option<(PipelineMemoryAccess, ImageLayout, ImageLayout)>,
            ),
        >,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        C: Command + 'static,
    {
        // TODO: see comment for the `is_poisoned` member in the struct
        assert!(
            !self.is_poisoned,
            "The builder has been put in an inconsistent state by a previous error"
        );

        // Note that we don't submit the command to the inner command buffer yet.
        let (latest_command_id, end) = {
            self.commands.push(Box::new(command));
            let latest_command_id = self.commands.len() - 1;
            let end = self.latest_render_pass_enter.unwrap_or(latest_command_id);
            (latest_command_id, end)
        };

        for (resource_ty, resource_name, resource) in resources {
            if let Some((memory, start_layout, end_layout)) = resource {
                // Anti-dumbness checks.
                debug_assert!(memory.exclusive || start_layout == end_layout);
                debug_assert!(memory.stages.supported_access().contains(&memory.access));

                match resource_ty {
                    KeyTy::Buffer(buffer) => {
                        debug_assert!(start_layout == ImageLayout::Undefined);
                        debug_assert!(end_layout == ImageLayout::Undefined);
                        self.add_buffer(buffer, resource_name, memory)?;
                    }
                    KeyTy::Image(image) => {
                        debug_assert!(end_layout != ImageLayout::Undefined);
                        debug_assert!(end_layout != ImageLayout::Preinitialized);
                        self.add_image(image, resource_name, memory, start_layout, end_layout)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn add_buffer(
        &mut self,
        buffer: Arc<dyn BufferAccess>,
        resource_name: Cow<'static, str>,
        memory: PipelineMemoryAccess,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        let latest_command_id = self.commands.len() - 1;
        let end = self.latest_render_pass_enter.unwrap_or(latest_command_id);

        let inner = buffer.inner();
        let range_map = self
            .buffers2
            .entry(inner.buffer.clone())
            .or_insert_with(|| [(0..inner.buffer.size(), None)].into_iter().collect());

        let range = inner.offset..inner.offset + buffer.size();
        range_map.split_at(&range.start);
        range_map.split_at(&range.end);

        for (range, state) in range_map.range_mut(&range) {
            match state {
                // Situation where this resource was used before in this command buffer.
                Some(state) => {
                    debug_assert!(state
                        .resource_uses
                        .iter()
                        .all(|resource_use| resource_use.command_index <= latest_command_id));

                    // Find out if we have a collision with the pending commands.
                    if memory.exclusive || state.memory.exclusive {
                        // Collision found between `latest_command_id` and `collision_cmd_id`.

                        // We now want to modify the current pipeline barrier in order to handle the
                        // collision. But since the pipeline barrier is going to be submitted before
                        // the flushed commands, it would be a mistake if `collision_cmd_id` hasn't
                        // been flushed yet.
                        let first_unflushed_cmd_id = self.first_unflushed;

                        if state.resource_uses.iter().any(|resource_use| {
                            resource_use.command_index >= first_unflushed_cmd_id
                        }) {
                            unsafe {
                                // Flush the pending barrier.
                                let barrier =
                                    replace(&mut self.pending_barrier, DependencyInfo::default());
                                self.inner.pipeline_barrier(barrier);

                                // Flush the commands if possible, or return an error if not possible.
                                {
                                    let start = self.first_unflushed;
                                    self.barriers.push(start); // Track inserted barriers

                                    if let Some(conflicting_use) = state
                                        .resource_uses
                                        .iter()
                                        .find(|resource_use| resource_use.command_index >= end)
                                    {
                                        // TODO: see comment for the `is_poisoned` member in the struct
                                        self.is_poisoned = true;

                                        let cmd2 = &self.commands[latest_command_id];

                                        return Err(SyncCommandBufferBuilderError::Conflict {
                                            command1_name: self.commands
                                                [conflicting_use.command_index]
                                                .name(),
                                            command1_param: conflicting_use.name.clone(),
                                            command1_offset: conflicting_use.command_index,

                                            command2_name: self.commands[latest_command_id].name(),
                                            command2_param: resource_name.clone(),
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

                        // Modify the pipeline barrier to handle the collision.
                        self.pending_barrier
                            .buffer_memory_barriers
                            .push(BufferMemoryBarrier {
                                source_stages: state.memory.stages,
                                source_access: state.memory.access,
                                destination_stages: memory.stages,
                                destination_access: memory.access,
                                range: range.clone(),
                                ..BufferMemoryBarrier::buffer(inner.buffer.clone())
                            });

                        state.resource_uses.push(BufferUse {
                            command_index: latest_command_id,
                            name: resource_name.clone(),
                        });

                        // Update state.
                        state.memory = memory;
                        state.exclusive_any = true;
                    } else {
                        // There is no collision. Simply merge the stages and accesses.
                        // TODO: what about simplifying the newly-constructed stages/accesses?
                        //       this would simplify the job of the driver, but is it worth it?
                        state.memory.stages |= memory.stages;
                        state.memory.access |= memory.access;
                    }
                }

                // Situation where this is the first time we use this resource in this command buffer.
                None => {
                    *state = Some(BufferState {
                        resource_uses: vec![BufferUse {
                            command_index: latest_command_id,
                            name: resource_name.clone(),
                        }],

                        memory: PipelineMemoryAccess {
                            stages: memory.stages,
                            access: memory.access,
                            exclusive: memory.exclusive,
                        },
                        exclusive_any: memory.exclusive,
                    });
                }
            }
        }

        self.buffers.push((buffer, memory));
        Ok(())
    }

    fn add_image(
        &mut self,
        image: Arc<dyn ImageAccess>,
        resource_name: Cow<'static, str>,
        memory: PipelineMemoryAccess,
        start_layout: ImageLayout,
        end_layout: ImageLayout,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        let latest_command_id = self.commands.len() - 1;
        let end = self.latest_render_pass_enter.unwrap_or(latest_command_id);

        let inner = image.inner();
        let range_map = self
            .images2
            .entry(inner.image.clone())
            .or_insert_with(|| [(0..inner.image.range_size(), None)].into_iter().collect());

        for range in inner.image.iter_ranges(
            image.format().aspects(),
            image.current_mip_levels_access(),
            image.current_array_layers_access(),
        ) {
            range_map.split_at(&range.start);
            range_map.split_at(&range.end);

            for (range, state) in range_map.range_mut(&range) {
                match state {
                    // Situation where this resource was used before in this command buffer.
                    Some(state) => {
                        debug_assert!(state
                            .resource_uses
                            .iter()
                            .all(|resource_use| resource_use.command_index <= latest_command_id));

                        // Find out if we have a collision with the pending commands.
                        if memory.exclusive
                            || state.memory.exclusive
                            || state.current_layout != start_layout
                        {
                            // Collision found between `latest_command_id` and `collision_cmd_id`.

                            // We now want to modify the current pipeline barrier in order to handle the
                            // collision. But since the pipeline barrier is going to be submitted before
                            // the flushed commands, it would be a mistake if `collision_cmd_id` hasn't
                            // been flushed yet.
                            let first_unflushed_cmd_id = self.first_unflushed;

                            if state.resource_uses.iter().any(|resource_use| {
                                resource_use.command_index >= first_unflushed_cmd_id
                            }) || state.current_layout != start_layout
                            {
                                unsafe {
                                    // Flush the pending barrier.
                                    let barrier = replace(
                                        &mut self.pending_barrier,
                                        DependencyInfo::default(),
                                    );
                                    self.inner.pipeline_barrier(barrier);

                                    // Flush the commands if possible, or return an error if not possible.
                                    {
                                        let start = self.first_unflushed;
                                        self.barriers.push(start); // Track inserted barriers

                                        if let Some(conflicting_use) = state
                                            .resource_uses
                                            .iter()
                                            .find(|resource_use| resource_use.command_index >= end)
                                        {
                                            // TODO: see comment for the `is_poisoned` member in the struct
                                            self.is_poisoned = true;

                                            let cmd2 = &self.commands[latest_command_id];

                                            return Err(SyncCommandBufferBuilderError::Conflict {
                                                command1_name: self.commands
                                                    [conflicting_use.command_index]
                                                    .name(),
                                                command1_param: conflicting_use.name.clone(),
                                                command1_offset: conflicting_use.command_index,

                                                command2_name: self.commands[latest_command_id]
                                                    .name(),
                                                command2_param: resource_name.clone(),
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

                            // Modify the pipeline barrier to handle the collision.
                            self.pending_barrier
                                .image_memory_barriers
                                .push(ImageMemoryBarrier {
                                    source_stages: state.memory.stages,
                                    source_access: state.memory.access,
                                    destination_stages: memory.stages,
                                    destination_access: memory.access,
                                    old_layout: state.current_layout,
                                    new_layout: start_layout,
                                    subresource_range: inner
                                        .image
                                        .range_to_subresources(range.clone()),
                                    ..ImageMemoryBarrier::image(inner.image.clone())
                                });

                            state.resource_uses.push(ImageUse {
                                command_index: latest_command_id,
                                name: resource_name.clone(),
                            });

                            // Update state.
                            state.memory = memory;
                            state.exclusive_any = true;
                            if memory.exclusive || end_layout != ImageLayout::Undefined {
                                // Only modify the layout in case of a write, because buffer operations
                                // pass `Undefined` for the layout. While a buffer write *must* set the
                                // layout to `Undefined`, a buffer read must not touch it.
                                state.current_layout = end_layout;
                            }
                        } else {
                            // There is no collision. Simply merge the stages and accesses.
                            // TODO: what about simplifying the newly-constructed stages/accesses?
                            //       this would simplify the job of the driver, but is it worth it?
                            state.memory.stages |= memory.stages;
                            state.memory.access |= memory.access;
                        }
                    }

                    // Situation where this is the first time we use this resource in this command buffer.
                    None => {
                        // We need to perform some tweaks if the initial layout requirement of the image
                        // is different from the first layout usage.
                        let mut actually_exclusive = memory.exclusive;
                        let mut actual_start_layout = start_layout;

                        if !self.is_secondary
                            && start_layout != ImageLayout::Undefined
                            && start_layout != ImageLayout::Preinitialized
                        {
                            let initial_layout_requirement = image.initial_layout_requirement();

                            // Checks if the image is initialized and transitions it
                            // if it isn't
                            let is_layout_initialized = image.is_layout_initialized();

                            if initial_layout_requirement != start_layout || !is_layout_initialized
                            {
                                // A layout transition is a write, so if we perform one, we need
                                // exclusive access.
                                actually_exclusive = true;

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
                                        if initial_layout_requirement != start_layout {
                                            actual_start_layout = initial_layout_requirement;
                                        }

                                        initial_layout_requirement
                                    } else {
                                        actual_start_layout = image.initial_layout();
                                        image.initial_layout()
                                    };
                                    self.pending_barrier.image_memory_barriers.push(
                                        ImageMemoryBarrier {
                                            source_stages: PipelineStages {
                                                bottom_of_pipe: true,
                                                ..PipelineStages::none()
                                            },
                                            source_access: AccessFlags::none(),
                                            destination_stages: memory.stages,
                                            destination_access: memory.access,
                                            old_layout: from_layout,
                                            new_layout: start_layout,
                                            subresource_range: inner
                                                .image
                                                .range_to_subresources(range.clone()),
                                            ..ImageMemoryBarrier::image(inner.image.clone())
                                        },
                                    );
                                    image.layout_initialized();
                                }
                            }
                        }

                        *state = Some(ImageState {
                            resource_uses: vec![ImageUse {
                                command_index: latest_command_id,
                                name: resource_name.clone(),
                            }],

                            memory: PipelineMemoryAccess {
                                stages: memory.stages,
                                access: memory.access,
                                exclusive: actually_exclusive,
                            },
                            exclusive_any: actually_exclusive,
                            initial_layout: actual_start_layout,
                            current_layout: end_layout,
                            final_layout: image.final_layout_requirement(),
                        });
                    }
                }
            }
        }

        self.images.push((image, memory, start_layout, end_layout));
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
            self.inner.pipeline_barrier(self.pending_barrier);
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
                let mut barrier = DependencyInfo::default();

                for (image, range_map) in self.images2.iter_mut() {
                    for (range, state) in range_map
                        .iter_mut()
                        .filter_map(|(range, state)| state.as_mut().map(|state| (range, state)))
                    {
                        if state.final_layout == state.current_layout {
                            continue;
                        }

                        barrier.image_memory_barriers.push(ImageMemoryBarrier {
                            source_stages: state.memory.stages,
                            source_access: state.memory.access,
                            destination_stages: PipelineStages {
                                top_of_pipe: true,
                                ..PipelineStages::none()
                            },
                            destination_access: AccessFlags::none(),
                            old_layout: state.current_layout,
                            new_layout: state.final_layout,
                            subresource_range: image.range_to_subresources(range.clone()),
                            ..ImageMemoryBarrier::image(image.clone())
                        });

                        state.exclusive_any = true;
                        state.current_layout = state.final_layout;
                    }
                }

                self.inner.pipeline_barrier(barrier);
            }
        }

        // Build the final resources states.
        let buffers2: HashMap<_, _> = self
            .buffers2
            .into_iter()
            .map(|(resource, range_map)| {
                let range_map = range_map
                    .into_iter()
                    .filter_map(|(range, state)| {
                        state.map(|state| {
                            let state = BufferFinalState {
                                resource_uses: state.resource_uses,
                                final_stages: state.memory.stages,
                                final_access: state.memory.access,
                                exclusive: state.exclusive_any,
                            };

                            (range, state)
                        })
                    })
                    .collect();

                (resource, range_map)
            })
            .collect();

        let images2: HashMap<_, _> = self
            .images2
            .into_iter()
            .map(|(resource, range_map)| {
                let range_map = range_map
                    .into_iter()
                    .filter_map(|(range, state)| {
                        state.map(|state| {
                            let state = ImageFinalState {
                                resource_uses: state.resource_uses,
                                final_stages: state.memory.stages,
                                final_access: state.memory.access,
                                exclusive: state.exclusive_any,
                                initial_layout: state.initial_layout,
                                final_layout: state.current_layout,
                            };

                            (range, state)
                        })
                    })
                    .collect();

                (resource, range_map)
            })
            .collect();

        Ok(SyncCommandBuffer {
            inner: self.inner.build()?,
            buffers: self.buffers,
            images: self.images,
            buffers2,
            images2,
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

// State of a resource during the building of the command buffer.
#[derive(Clone, PartialEq, Eq)]
struct BufferState {
    // Lists every use of the resource.
    resource_uses: Vec<BufferUse>,

    // Memory access of the command that last used this resource.
    memory: PipelineMemoryAccess,

    // True if the resource was used in exclusive mode at any point during the building of the
    // command buffer. Also true if an image layout transition or queue transfer has been performed.
    exclusive_any: bool,
}

// State of a resource during the building of the command buffer.
#[derive(Clone, PartialEq, Eq)]
struct ImageState {
    // Lists every use of the resource.
    resource_uses: Vec<ImageUse>,

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

    // The layout to transition to at the end of the command buffer.
    final_layout: ImageLayout,
}

/// Holds the current binding and setting state.
#[derive(Default)]
pub(in crate::command_buffer) struct CurrentState {
    pub(in crate::command_buffer) descriptor_sets: HashMap<PipelineBindPoint, DescriptorSetState>,
    pub(in crate::command_buffer) index_buffer: Option<(Arc<dyn BufferAccess>, IndexType)>,
    pub(in crate::command_buffer) pipeline_compute: Option<Arc<ComputePipeline>>,
    pub(in crate::command_buffer) pipeline_graphics: Option<Arc<GraphicsPipeline>>,
    pub(in crate::command_buffer) vertex_buffers: HashMap<u32, Arc<dyn BufferAccess>>,

    pub(in crate::command_buffer) push_constants: RangeSet<u32>,
    pub(in crate::command_buffer) push_constants_pipeline_layout: Option<Arc<PipelineLayout>>,

    pub(in crate::command_buffer) blend_constants: Option<[f32; 4]>,
    pub(in crate::command_buffer) color_write_enable: Option<SmallVec<[bool; 4]>>,
    pub(in crate::command_buffer) cull_mode: Option<CullMode>,
    pub(in crate::command_buffer) depth_bias: Option<DepthBias>,
    pub(in crate::command_buffer) depth_bias_enable: Option<bool>,
    pub(in crate::command_buffer) depth_bounds: Option<(f32, f32)>,
    pub(in crate::command_buffer) depth_bounds_test_enable: Option<bool>,
    pub(in crate::command_buffer) depth_compare_op: Option<CompareOp>,
    pub(in crate::command_buffer) depth_test_enable: Option<bool>,
    pub(in crate::command_buffer) depth_write_enable: Option<bool>,
    pub(in crate::command_buffer) discard_rectangle: HashMap<u32, Scissor>,
    pub(in crate::command_buffer) front_face: Option<FrontFace>,
    pub(in crate::command_buffer) line_stipple: Option<LineStipple>,
    pub(in crate::command_buffer) line_width: Option<f32>,
    pub(in crate::command_buffer) logic_op: Option<LogicOp>,
    pub(in crate::command_buffer) patch_control_points: Option<u32>,
    pub(in crate::command_buffer) primitive_restart_enable: Option<bool>,
    pub(in crate::command_buffer) primitive_topology: Option<PrimitiveTopology>,
    pub(in crate::command_buffer) rasterizer_discard_enable: Option<bool>,
    pub(in crate::command_buffer) scissor: HashMap<u32, Scissor>,
    pub(in crate::command_buffer) scissor_with_count: Option<SmallVec<[Scissor; 2]>>,
    pub(in crate::command_buffer) stencil_compare_mask: StencilStateDynamic,
    pub(in crate::command_buffer) stencil_op: StencilOpStateDynamic,
    pub(in crate::command_buffer) stencil_reference: StencilStateDynamic,
    pub(in crate::command_buffer) stencil_test_enable: Option<bool>,
    pub(in crate::command_buffer) stencil_write_mask: StencilStateDynamic,
    pub(in crate::command_buffer) viewport: HashMap<u32, Viewport>,
    pub(in crate::command_buffer) viewport_with_count: Option<SmallVec<[Viewport; 2]>>,
}

impl CurrentState {
    pub(in crate::command_buffer) fn reset_dynamic_states(
        &mut self,
        states: impl IntoIterator<Item = DynamicState>,
    ) {
        for state in states {
            match state {
                DynamicState::BlendConstants => self.blend_constants = None,
                DynamicState::ColorWriteEnable => self.color_write_enable = None,
                DynamicState::CullMode => self.cull_mode = None,
                DynamicState::DepthBias => self.depth_bias = None,
                DynamicState::DepthBiasEnable => self.depth_bias_enable = None,
                DynamicState::DepthBounds => self.depth_bounds = None,
                DynamicState::DepthBoundsTestEnable => self.depth_bounds_test_enable = None,
                DynamicState::DepthCompareOp => self.depth_compare_op = None,
                DynamicState::DepthTestEnable => self.depth_test_enable = None,
                DynamicState::DepthWriteEnable => self.depth_write_enable = None,
                DynamicState::DiscardRectangle => self.discard_rectangle.clear(),
                DynamicState::ExclusiveScissor => (), // TODO;
                DynamicState::FragmentShadingRate => (), // TODO:
                DynamicState::FrontFace => self.front_face = None,
                DynamicState::LineStipple => self.line_stipple = None,
                DynamicState::LineWidth => self.line_width = None,
                DynamicState::LogicOp => self.logic_op = None,
                DynamicState::PatchControlPoints => self.patch_control_points = None,
                DynamicState::PrimitiveRestartEnable => self.primitive_restart_enable = None,
                DynamicState::PrimitiveTopology => self.primitive_topology = None,
                DynamicState::RasterizerDiscardEnable => self.rasterizer_discard_enable = None,
                DynamicState::RayTracingPipelineStackSize => (), // TODO:
                DynamicState::SampleLocations => (),             // TODO:
                DynamicState::Scissor => self.scissor.clear(),
                DynamicState::ScissorWithCount => self.scissor_with_count = None,
                DynamicState::StencilCompareMask => self.stencil_compare_mask = Default::default(),
                DynamicState::StencilOp => self.stencil_op = Default::default(),
                DynamicState::StencilReference => self.stencil_reference = Default::default(),
                DynamicState::StencilTestEnable => self.stencil_test_enable = None,
                DynamicState::StencilWriteMask => self.stencil_write_mask = Default::default(),
                DynamicState::VertexInput => (), // TODO:
                DynamicState::VertexInputBindingStride => (), // TODO:
                DynamicState::Viewport => self.viewport.clear(),
                DynamicState::ViewportCoarseSampleOrder => (), // TODO:
                DynamicState::ViewportShadingRatePalette => (), // TODO:
                DynamicState::ViewportWScaling => (),          // TODO:
                DynamicState::ViewportWithCount => self.viewport_with_count = None,
            }
        }
    }

    pub(in crate::command_buffer) fn invalidate_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        first_set: u32,
        num_descriptor_sets: u32,
    ) -> &mut DescriptorSetState {
        match self.descriptor_sets.entry(pipeline_bind_point) {
            Entry::Vacant(entry) => entry.insert(DescriptorSetState {
                descriptor_sets: Default::default(),
                pipeline_layout,
            }),
            Entry::Occupied(entry) => {
                let state = entry.into_mut();

                let invalidate_from = if state.pipeline_layout.internal_object()
                    == pipeline_layout.internal_object()
                {
                    // If we're still using the exact same layout, then of course it's compatible.
                    None
                } else if state.pipeline_layout.push_constant_ranges()
                    != pipeline_layout.push_constant_ranges()
                {
                    // If the push constant ranges don't match,
                    // all bound descriptor sets are disturbed.
                    Some(0)
                } else {
                    // Find the first descriptor set layout in the current pipeline layout that
                    // isn't compatible with the corresponding set in the new pipeline layout.
                    // If an incompatible set was found, all bound sets from that slot onwards will
                    // be disturbed.
                    let current_layouts = state.pipeline_layout.set_layouts();
                    let new_layouts = pipeline_layout.set_layouts();
                    let max = (current_layouts.len() as u32).min(first_set + num_descriptor_sets);
                    (0..max).find(|&num| {
                        let num = num as usize;
                        !current_layouts[num].is_compatible_with(&new_layouts[num])
                    })
                };

                if let Some(invalidate_from) = invalidate_from {
                    // Remove disturbed sets and set new pipeline layout.
                    state
                        .descriptor_sets
                        .retain(|&num, _| num < invalidate_from);
                    state.pipeline_layout = pipeline_layout;
                } else if (first_set + num_descriptor_sets) as usize
                    >= state.pipeline_layout.set_layouts().len()
                {
                    // New layout is a superset of the old one.
                    state.pipeline_layout = pipeline_layout;
                }

                state
            }
        }
    }
}

pub(in crate::command_buffer) struct DescriptorSetState {
    pub(in crate::command_buffer) descriptor_sets: HashMap<u32, SetOrPush>,
    pub(in crate::command_buffer) pipeline_layout: Arc<PipelineLayout>,
}

#[derive(Clone)]
pub enum SetOrPush {
    Set(DescriptorSetWithOffsets),
    Push(DescriptorSetResources),
}

impl SetOrPush {
    pub fn resources(&self) -> &DescriptorSetResources {
        match self {
            Self::Set(set) => set.as_ref().0.resources(),
            Self::Push(resources) => resources,
        }
    }
}

/// Allows you to retrieve the current state of a command buffer builder.
#[derive(Clone, Copy)]
pub struct CommandBufferState<'a> {
    current_state: &'a CurrentState,
}

impl<'a> CommandBufferState<'a> {
    /// Returns the descriptor set currently bound to a given set number, or `None` if nothing has
    /// been bound yet.
    #[inline]
    pub fn descriptor_set(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        set_num: u32,
    ) -> Option<&'a SetOrPush> {
        self.current_state
            .descriptor_sets
            .get(&pipeline_bind_point)
            .and_then(|state| state.descriptor_sets.get(&set_num))
    }

    /// Returns the pipeline layout that describes all currently bound descriptor sets.
    ///
    /// This can be the layout used to perform the last bind operation, but it can also be the
    /// layout of an earlier bind if it was compatible with more recent binds.
    #[inline]
    pub fn descriptor_sets_pipeline_layout(
        &self,
        pipeline_bind_point: PipelineBindPoint,
    ) -> Option<&'a Arc<PipelineLayout>> {
        self.current_state
            .descriptor_sets
            .get(&pipeline_bind_point)
            .map(|state| &state.pipeline_layout)
    }

    /// Returns the index buffer currently bound, or `None` if nothing has been bound yet.
    #[inline]
    pub fn index_buffer(&self) -> Option<(&'a Arc<dyn BufferAccess>, IndexType)> {
        self.current_state
            .index_buffer
            .as_ref()
            .map(|(b, i)| (b, *i))
    }

    /// Returns the compute pipeline currently bound, or `None` if nothing has been bound yet.
    #[inline]
    pub fn pipeline_compute(&self) -> Option<&'a Arc<ComputePipeline>> {
        self.current_state.pipeline_compute.as_ref()
    }

    /// Returns the graphics pipeline currently bound, or `None` if nothing has been bound yet.
    #[inline]
    pub fn pipeline_graphics(&self) -> Option<&'a Arc<GraphicsPipeline>> {
        self.current_state.pipeline_graphics.as_ref()
    }

    /// Returns the vertex buffer currently bound to a given binding slot number, or `None` if
    /// nothing has been bound yet.
    #[inline]
    pub fn vertex_buffer(&self, binding_num: u32) -> Option<&'a Arc<dyn BufferAccess>> {
        self.current_state.vertex_buffers.get(&binding_num)
    }

    /// Returns a set containing push constant bytes that have been set.
    #[inline]
    pub fn push_constants(&self) -> &'a RangeSet<u32> {
        &self.current_state.push_constants
    }

    /// Returns the pipeline layout that describes the current push constants.
    ///
    /// This is the layout used to perform the last push constant write operation.
    #[inline]
    pub fn push_constants_pipeline_layout(&self) -> Option<&'a Arc<PipelineLayout>> {
        self.current_state.push_constants_pipeline_layout.as_ref()
    }

    /// Returns the current blend constants, or `None` if nothing has been set yet.
    #[inline]
    pub fn blend_constants(&self) -> Option<[f32; 4]> {
        self.current_state.blend_constants
    }

    /// Returns the current color write enable settings, or `None` if nothing has been set yet.
    #[inline]
    pub fn color_write_enable(&self) -> Option<&'a [bool]> {
        self.current_state
            .color_write_enable
            .as_ref()
            .map(|x| x.as_slice())
    }

    /// Returns the current cull mode, or `None` if nothing has been set yet.
    #[inline]
    pub fn cull_mode(&self) -> Option<CullMode> {
        self.current_state.cull_mode
    }

    /// Returns the current depth bias settings, or `None` if nothing has been set yet.
    #[inline]
    pub fn depth_bias(&self) -> Option<DepthBias> {
        self.current_state.depth_bias
    }

    /// Returns whether depth bias is enabled, or `None` if nothing has been set yet.
    #[inline]
    pub fn depth_bias_enable(&self) -> Option<bool> {
        self.current_state.depth_bias_enable
    }

    /// Returns the current depth bounds settings, or `None` if nothing has been set yet.
    #[inline]
    pub fn depth_bounds(&self) -> Option<(f32, f32)> {
        self.current_state.depth_bounds
    }

    /// Returns whether depth bound testing is enabled, or `None` if nothing has been set yet.
    #[inline]
    pub fn depth_bounds_test_enable(&self) -> Option<bool> {
        self.current_state.depth_bias_enable
    }

    /// Returns the current depth compare op, or `None` if nothing has been set yet.
    #[inline]
    pub fn depth_compare_op(&self) -> Option<CompareOp> {
        self.current_state.depth_compare_op
    }

    /// Returns whether depth testing is enabled, or `None` if nothing has been set yet.
    #[inline]
    pub fn depth_test_enable(&self) -> Option<bool> {
        self.current_state.depth_test_enable
    }

    /// Returns whether depth write is enabled, or `None` if nothing has been set yet.
    #[inline]
    pub fn depth_write_enable(&self) -> Option<bool> {
        self.current_state.depth_write_enable
    }

    /// Returns the current discard rectangles, or `None` if nothing has been set yet.
    #[inline]
    pub fn discard_rectangle(&self, num: u32) -> Option<&'a Scissor> {
        self.current_state.discard_rectangle.get(&num)
    }

    /// Returns the current front face, or `None` if nothing has been set yet.
    #[inline]
    pub fn front_face(&self) -> Option<FrontFace> {
        self.current_state.front_face
    }

    /// Returns the current line stipple settings, or `None` if nothing has been set yet.
    #[inline]
    pub fn line_stipple(&self) -> Option<LineStipple> {
        self.current_state.line_stipple
    }

    /// Returns the current line width, or `None` if nothing has been set yet.
    #[inline]
    pub fn line_width(&self) -> Option<f32> {
        self.current_state.line_width
    }

    /// Returns the current logic op, or `None` if nothing has been set yet.
    #[inline]
    pub fn logic_op(&self) -> Option<LogicOp> {
        self.current_state.logic_op
    }

    /// Returns the current number of patch control points, or `None` if nothing has been set yet.
    #[inline]
    pub fn patch_control_points(&self) -> Option<u32> {
        self.current_state.patch_control_points
    }

    /// Returns whether primitive restart is enabled, or `None` if nothing has been set yet.
    #[inline]
    pub fn primitive_restart_enable(&self) -> Option<bool> {
        self.current_state.primitive_restart_enable
    }

    /// Returns the current primitive topology, or `None` if nothing has been set yet.
    #[inline]
    pub fn primitive_topology(&self) -> Option<PrimitiveTopology> {
        self.current_state.primitive_topology
    }

    /// Returns whether rasterizer discard is enabled, or `None` if nothing has been set yet.
    #[inline]
    pub fn rasterizer_discard_enable(&self) -> Option<bool> {
        self.current_state.rasterizer_discard_enable
    }

    /// Returns the current scissor for a given viewport slot, or `None` if nothing has been set yet.
    #[inline]
    pub fn scissor(&self, num: u32) -> Option<&'a Scissor> {
        self.current_state.scissor.get(&num)
    }

    /// Returns the current viewport-with-count settings, or `None` if nothing has been set yet.
    #[inline]
    pub fn scissor_with_count(&self) -> Option<&'a [Scissor]> {
        self.current_state
            .scissor_with_count
            .as_ref()
            .map(|x| x.as_slice())
    }

    /// Returns the current stencil compare masks.
    #[inline]
    pub fn stencil_compare_mask(&self) -> StencilStateDynamic {
        self.current_state.stencil_compare_mask
    }

    /// Returns the current stencil ops.
    #[inline]
    pub fn stencil_op(&self) -> StencilOpStateDynamic {
        self.current_state.stencil_op
    }

    /// Returns the current stencil references.
    #[inline]
    pub fn stencil_reference(&self) -> StencilStateDynamic {
        self.current_state.stencil_reference
    }

    /// Returns whether stencil testing is enabled, or `None` if nothing has been set yet.
    #[inline]
    pub fn stencil_test_enable(&self) -> Option<bool> {
        self.current_state.stencil_test_enable
    }

    /// Returns the current stencil write masks.
    #[inline]
    pub fn stencil_write_mask(&self) -> StencilStateDynamic {
        self.current_state.stencil_write_mask
    }

    /// Returns the current viewport for a given viewport slot, or `None` if nothing has been set yet.
    #[inline]
    pub fn viewport(&self, num: u32) -> Option<&'a Viewport> {
        self.current_state.viewport.get(&num)
    }

    /// Returns the current viewport-with-count settings, or `None` if nothing has been set yet.
    #[inline]
    pub fn viewport_with_count(&self) -> Option<&'a [Viewport]> {
        self.current_state
            .viewport_with_count
            .as_ref()
            .map(|x| x.as_slice())
    }
}

/// Holds the current stencil state of a command buffer builder.
#[derive(Clone, Copy, Debug, Default)]
pub struct StencilStateDynamic {
    pub front: Option<u32>,
    pub back: Option<u32>,
}

/// Holds the current per-face stencil op state of a command buffer builder.
#[derive(Clone, Copy, Debug, Default)]
pub struct StencilOpStateDynamic {
    pub front: Option<StencilOps>,
    pub back: Option<StencilOps>,
}
