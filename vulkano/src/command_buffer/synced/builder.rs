// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{Command, Resource, SyncCommandBuffer};
pub use crate::command_buffer::commands::{
    bind_push::{
        SyncCommandBufferBuilderBindDescriptorSets, SyncCommandBufferBuilderBindVertexBuffer,
    },
    secondary::SyncCommandBufferBuilderExecuteCommands,
};
use crate::{
    buffer::{sys::Buffer, BufferAccess},
    command_buffer::{
        pool::CommandPoolAlloc,
        synced::{BufferUse, ImageUse},
        sys::{CommandBufferBeginInfo, UnsafeCommandBufferBuilder},
        CommandBufferBufferRangeUsage, CommandBufferBufferUsage, CommandBufferExecError,
        CommandBufferImageRangeUsage, CommandBufferImageUsage, CommandBufferLevel,
        CommandBufferResourcesUsage, FirstResourceUse,
    },
    descriptor_set::{DescriptorSetResources, DescriptorSetWithOffsets},
    device::{Device, DeviceOwned},
    image::{sys::Image, ImageAccess, ImageLayout, ImageSubresourceRange},
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
    range_map::RangeMap,
    range_set::RangeSet,
    sync::{
        AccessFlags, BufferMemoryBarrier, DependencyInfo, ImageMemoryBarrier, PipelineMemoryAccess,
        PipelineStages,
    },
    DeviceSize, OomError, VulkanObject,
};
use ahash::HashMap;
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    collections::hash_map::Entry,
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    ops::{Range, RangeInclusive},
    sync::Arc,
};

/// Wrapper around `UnsafeCommandBufferBuilder` that handles synchronization for you.
///
/// Each method of the `UnsafeCommandBufferBuilder` has an equivalent in this wrapper, except
/// for `pipeline_layout` which is automatically handled. This wrapper automatically builds
/// pipeline barriers, keeps used resources alive and implements the `CommandBuffer` trait.
///
/// If this builder finds out that a command isn't valid because of synchronization reasons (eg.
/// trying to copy from a buffer to an image which share the same memory), then an error is
/// returned.
/// Note that all methods are still unsafe, because this builder doesn't check the validity of
/// the commands except for synchronization purposes. The builder may panic if you pass invalid
/// commands.
pub struct SyncCommandBufferBuilder {
    inner: UnsafeCommandBufferBuilder,
    level: CommandBufferLevel,

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
    buffers2: HashMap<Arc<Buffer>, RangeMap<DeviceSize, BufferState>>,
    images2: HashMap<Arc<Image>, RangeMap<DeviceSize, ImageState>>,

    // Resources and their accesses. Used for executing secondary command buffers in a primary.
    buffers: Vec<(
        Arc<dyn BufferAccess>,
        Range<DeviceSize>,
        PipelineMemoryAccess,
    )>,
    images: Vec<(
        Arc<dyn ImageAccess>,
        ImageSubresourceRange,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
    )>,

    // Current binding/setting state.
    pub(in crate::command_buffer) current_state: CurrentState,
}

impl SyncCommandBufferBuilder {
    /// Builds a new `SyncCommandBufferBuilder`. The parameters are the same as the
    /// `UnsafeCommandBufferBuilder::new` function.
    ///
    /// # Safety
    ///
    /// See `UnsafeCommandBufferBuilder::new()`.
    #[inline]
    pub unsafe fn new(
        pool_alloc: &CommandPoolAlloc,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<SyncCommandBufferBuilder, OomError> {
        let level = pool_alloc.level();
        let inside_render_pass = level == CommandBufferLevel::Secondary
            && begin_info
                .inheritance_info
                .as_ref()
                .unwrap()
                .render_pass
                .is_some();
        let inner = UnsafeCommandBufferBuilder::new(pool_alloc, begin_info)?;

        Ok(SyncCommandBufferBuilder::from_unsafe_cmd(
            inner,
            level,
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
        inner: UnsafeCommandBufferBuilder,
        level: CommandBufferLevel,
        inside_render_pass: bool,
    ) -> SyncCommandBufferBuilder {
        let latest_render_pass_enter = if inside_render_pass { Some(0) } else { None };

        SyncCommandBufferBuilder {
            inner,
            level,
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
        }
    }

    /// Returns the binding/setting state.
    #[inline]
    pub fn state(&self) -> CommandBufferBuilderState<'_> {
        CommandBufferBuilderState {
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

    pub(in crate::command_buffer) fn check_resource_conflicts(
        &self,
        resource: &(Cow<'static, str>, Resource),
    ) -> Result<(), SyncCommandBufferBuilderError> {
        let (resource_name, resource) = resource;

        match *resource {
            Resource::Buffer {
                ref buffer,
                ref range,
                ref memory,
            } => {
                debug_assert!(memory.stages.supported_access().contains(memory.access));

                if let Some(conflicting_use) =
                    self.find_buffer_conflict(buffer, range.clone(), memory)
                {
                    return Err(SyncCommandBufferBuilderError::Conflict {
                        command_param: resource_name.clone(),
                        previous_command_name: self.commands[conflicting_use.command_index].name(),
                        previous_command_offset: conflicting_use.command_index,
                        previous_command_param: conflicting_use.name.clone(),
                    });
                }
            }
            Resource::Image {
                ref image,
                ref subresource_range,
                ref memory,
                start_layout,
                end_layout,
            } => {
                debug_assert!(memory.exclusive || start_layout == end_layout);
                debug_assert!(memory.stages.supported_access().contains(memory.access));
                debug_assert!(end_layout != ImageLayout::Undefined);
                debug_assert!(end_layout != ImageLayout::Preinitialized);

                if let Some(conflicting_use) = self.find_image_conflict(
                    image,
                    subresource_range.clone(),
                    memory,
                    start_layout,
                    end_layout,
                ) {
                    return Err(SyncCommandBufferBuilderError::Conflict {
                        command_param: resource_name.clone(),
                        previous_command_name: self.commands[conflicting_use.command_index].name(),
                        previous_command_offset: conflicting_use.command_index,
                        previous_command_param: conflicting_use.name.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    fn find_buffer_conflict(
        &self,
        buffer: &dyn BufferAccess,
        mut range: Range<DeviceSize>,
        memory: &PipelineMemoryAccess,
    ) -> Option<&BufferUse> {
        // Barriers work differently in render passes, so if we're in one, we can only insert a
        // barrier before the start of the render pass.
        let last_allowed_barrier_index =
            self.latest_render_pass_enter.unwrap_or(self.commands.len());

        let inner = buffer.inner();
        range.start += inner.offset;
        range.end += inner.offset;

        let range_map = self.buffers2.get(inner.buffer)?;

        for (_range, state) in range_map
            .range(&range)
            .filter(|(_range, state)| !state.resource_uses.is_empty())
        {
            debug_assert!(state
                .resource_uses
                .iter()
                .all(|resource_use| resource_use.command_index <= self.commands.len()));

            if memory.exclusive || state.memory.exclusive {
                // If there is a resource use at a position beyond where we can insert a
                // barrier, then there is an unsolvable conflict.
                if let Some(conflicting_use) = state
                    .resource_uses
                    .iter()
                    .find(|resource_use| resource_use.command_index >= last_allowed_barrier_index)
                {
                    return Some(conflicting_use);
                }
            }
        }

        None
    }

    fn find_image_conflict(
        &self,
        image: &dyn ImageAccess,
        mut subresource_range: ImageSubresourceRange,
        memory: &PipelineMemoryAccess,
        start_layout: ImageLayout,
        _end_layout: ImageLayout,
    ) -> Option<&ImageUse> {
        // Barriers work differently in render passes, so if we're in one, we can only insert a
        // barrier before the start of the render pass.
        let last_allowed_barrier_index =
            self.latest_render_pass_enter.unwrap_or(self.commands.len());

        let inner = image.inner();
        subresource_range.array_layers.start += inner.first_layer;
        subresource_range.array_layers.end += inner.first_layer;
        subresource_range.mip_levels.start += inner.first_mipmap_level;
        subresource_range.mip_levels.end += inner.first_mipmap_level;

        let range_map = self.images2.get(inner.image)?;

        for range in inner.image.iter_ranges(subresource_range) {
            for (_range, state) in range_map
                .range(&range)
                .filter(|(_range, state)| !state.resource_uses.is_empty())
            {
                debug_assert!(state
                    .resource_uses
                    .iter()
                    .all(|resource_use| resource_use.command_index <= self.commands.len()));

                // If the command expects the image to be undefined, then we can't
                // transition it, so use the current layout for both old and new layout.
                let start_layout = if start_layout == ImageLayout::Undefined {
                    state.current_layout
                } else {
                    start_layout
                };

                if memory.exclusive
                    || state.memory.exclusive
                    || state.current_layout != start_layout
                {
                    // If there is a resource use at a position beyond where we can insert a
                    // barrier, then there is an unsolvable conflict.
                    if let Some(conflicting_use) = state.resource_uses.iter().find(|resource_use| {
                        resource_use.command_index >= last_allowed_barrier_index
                    }) {
                        return Some(conflicting_use);
                    }
                }
            }
        }

        None
    }

    /// Adds a command to be processed by the builder.
    ///
    /// The `resources` argument should contain each buffer or image used by the command.
    /// The function will take care of handling the pipeline barrier or flushing.
    ///
    /// - The index of the resource within the `resources` slice maps to the resource accessed
    ///   through `Command::buffer(..)` or `Command::image(..)`.
    /// - `PipelineMemoryAccess` must match the way the resource has been used.
    /// - `start_layout` and `end_layout` designate the image layout that the image is expected to
    ///   be in when the command starts, and the image layout that the image will be transitioned to
    ///   during the command. When it comes to buffers, you should pass `Undefined` for both.
    pub(in crate::command_buffer) fn add_resource(
        &mut self,
        resource: (Cow<'static, str>, Resource),
    ) {
        let (resource_name, resource) = resource;

        match resource {
            Resource::Buffer {
                buffer,
                range,
                memory,
            } => {
                self.add_buffer(resource_name, buffer, range, memory);
            }
            Resource::Image {
                image,
                subresource_range,
                memory,
                start_layout,
                end_layout,
            } => {
                self.add_image(
                    resource_name,
                    image,
                    subresource_range,
                    memory,
                    start_layout,
                    end_layout,
                );
            }
        }
    }

    fn add_buffer(
        &mut self,
        resource_name: Cow<'static, str>,
        buffer: Arc<dyn BufferAccess>,
        mut range: Range<DeviceSize>,
        memory: PipelineMemoryAccess,
    ) {
        self.buffers.push((buffer.clone(), range.clone(), memory));

        // Barriers work differently in render passes, so if we're in one, we can only insert a
        // barrier before the start of the render pass.
        let last_allowed_barrier_index = self
            .latest_render_pass_enter
            .unwrap_or(self.commands.len() - 1);

        let inner = buffer.inner();
        range.start += inner.offset;
        range.end += inner.offset;

        let range_map = self
            .buffers2
            .entry(inner.buffer.clone())
            .or_insert_with(|| {
                [(
                    0..inner.buffer.size(),
                    BufferState {
                        resource_uses: Vec::new(),
                        memory: PipelineMemoryAccess::default(),
                        exclusive_any: false,
                    },
                )]
                .into_iter()
                .collect()
            });
        range_map.split_at(&range.start);
        range_map.split_at(&range.end);

        for (range, state) in range_map.range_mut(&range) {
            if state.resource_uses.is_empty() {
                // This is the first time we use this resource range in this command buffer.
                state.resource_uses.push(BufferUse {
                    command_index: self.commands.len() - 1,
                    name: resource_name.clone(),
                });
                state.memory = PipelineMemoryAccess {
                    stages: memory.stages,
                    access: memory.access,
                    exclusive: memory.exclusive,
                };
                state.exclusive_any = memory.exclusive;

                match self.level {
                    CommandBufferLevel::Primary => {
                        // To be safe, we insert a barrier for all stages and accesses before
                        // the first use, so that there are no hazards with any command buffer
                        // that was previously submitted to the same queue.
                        // This is rather overkill, but since command buffers don't know what
                        // will come before them, it's the only thing that works for now.
                        // TODO: come up with something better
                        let barrier = BufferMemoryBarrier {
                            src_stages: PipelineStages::ALL_COMMANDS,
                            src_access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                            dst_stages: PipelineStages::ALL_COMMANDS,
                            dst_access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                            range: range.clone(),
                            ..BufferMemoryBarrier::buffer(inner.buffer.clone())
                        };

                        self.pending_barrier.buffer_memory_barriers.push(barrier);
                    }
                    CommandBufferLevel::Secondary => (),
                }
            } else {
                // This resource range was used before in this command buffer.

                // Find out if we have a collision with the pending commands.
                if memory.exclusive || state.memory.exclusive {
                    // Collision found between `latest_command_id` and `collision_cmd_id`.

                    // We now want to modify the current pipeline barrier in order to handle the
                    // collision. But since the pipeline barrier is going to be submitted before
                    // the flushed commands, it would be a mistake if `collision_cmd_id` hasn't
                    // been flushed yet.
                    if state
                        .resource_uses
                        .iter()
                        .any(|resource_use| resource_use.command_index >= self.first_unflushed)
                    {
                        unsafe {
                            // Flush the pending barrier.
                            self.inner.pipeline_barrier(&self.pending_barrier);
                            self.pending_barrier.clear();
                            self.barriers.push(self.first_unflushed); // Track inserted barriers

                            for command in
                                &mut self.commands[self.first_unflushed..last_allowed_barrier_index]
                            {
                                command.send(&mut self.inner);
                            }

                            self.first_unflushed = last_allowed_barrier_index;
                        }
                    }

                    // Modify the pipeline barrier to handle the collision.
                    self.pending_barrier
                        .buffer_memory_barriers
                        .push(BufferMemoryBarrier {
                            src_stages: state.memory.stages,
                            src_access: state.memory.access,
                            dst_stages: memory.stages,
                            dst_access: memory.access,
                            range: range.clone(),
                            ..BufferMemoryBarrier::buffer(inner.buffer.clone())
                        });

                    // Update state.
                    state.memory = memory;
                    state.exclusive_any = true;
                } else {
                    // There is no collision. Simply merge the stages and accesses.
                    state.memory.stages |= memory.stages;
                    state.memory.access |= memory.access;
                }

                state.resource_uses.push(BufferUse {
                    command_index: self.commands.len() - 1,
                    name: resource_name.clone(),
                });
            }
        }
    }

    fn add_image(
        &mut self,
        resource_name: Cow<'static, str>,
        image: Arc<dyn ImageAccess>,
        mut subresource_range: ImageSubresourceRange,
        memory: PipelineMemoryAccess,
        start_layout: ImageLayout,
        end_layout: ImageLayout,
    ) {
        self.images.push((
            image.clone(),
            subresource_range.clone(),
            memory,
            start_layout,
            end_layout,
        ));

        // Barriers work differently in render passes, so if we're in one, we can only insert a
        // barrier before the start of the render pass.
        let last_allowed_barrier_index = self
            .latest_render_pass_enter
            .unwrap_or(self.commands.len() - 1);

        let inner = image.inner();
        subresource_range.array_layers.start += inner.first_layer;
        subresource_range.array_layers.end += inner.first_layer;
        subresource_range.mip_levels.start += inner.first_mipmap_level;
        subresource_range.mip_levels.end += inner.first_mipmap_level;

        let range_map = self.images2.entry(inner.image.clone()).or_insert_with(|| {
            [(
                0..inner.image.range_size(),
                match self.level {
                    CommandBufferLevel::Primary => {
                        // In a primary command buffer, the initial layout is determined
                        // by the image.
                        let initial_layout = if !image.is_layout_initialized() {
                            unsafe {
                                image.layout_initialized();
                            }

                            image.initial_layout()
                        } else {
                            image.initial_layout_requirement()
                        };

                        ImageState {
                            resource_uses: Vec::new(),
                            memory: PipelineMemoryAccess::default(),
                            exclusive_any: false,
                            initial_layout,
                            current_layout: initial_layout,
                            final_layout: image.final_layout_requirement(),
                        }
                    }
                    CommandBufferLevel::Secondary => {
                        // In a secondary command buffer, the initial layout is the layout
                        // of the first use.
                        ImageState {
                            resource_uses: Vec::new(),
                            memory: PipelineMemoryAccess::default(),
                            exclusive_any: false,
                            initial_layout: ImageLayout::Undefined,
                            current_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::Undefined,
                        }
                    }
                },
            )]
            .into_iter()
            .collect()
        });

        for range in inner.image.iter_ranges(subresource_range) {
            range_map.split_at(&range.start);
            range_map.split_at(&range.end);

            for (range, state) in range_map.range_mut(&range) {
                if state.resource_uses.is_empty() {
                    // This is the first time we use this resource range in this command buffer.

                    debug_assert_eq!(state.initial_layout, state.current_layout);

                    state.resource_uses.push(ImageUse {
                        command_index: self.commands.len() - 1,
                        name: resource_name.clone(),
                    });
                    state.memory = PipelineMemoryAccess {
                        stages: memory.stages,
                        access: memory.access,
                        exclusive: memory.exclusive,
                    };
                    state.exclusive_any = memory.exclusive;
                    state.current_layout = end_layout;

                    match self.level {
                        CommandBufferLevel::Primary => {
                            // To be safe, we insert a barrier for all stages and accesses before
                            // the first use, so that there are no hazards with any command buffer
                            // that was previously submitted to the same queue.
                            // This is rather overkill, but since command buffers don't know what
                            // will come before them, it's the only thing that works for now.
                            // TODO: come up with something better
                            let mut barrier = ImageMemoryBarrier {
                                src_stages: PipelineStages::ALL_COMMANDS,
                                src_access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                                dst_stages: PipelineStages::ALL_COMMANDS,
                                dst_access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                                old_layout: state.initial_layout,
                                new_layout: state.initial_layout,
                                subresource_range: inner.image.range_to_subresources(range.clone()),
                                ..ImageMemoryBarrier::image(inner.image.clone())
                            };

                            if state.initial_layout != start_layout {
                                match start_layout {
                                    ImageLayout::Undefined => {
                                        // We can't transition to `Undefined`,
                                        // but since this means that the command doesn't need any
                                        // particular layout, we do nothing.
                                    }
                                    ImageLayout::Preinitialized => {
                                        // We can't transition to `Preinitialized`,
                                        // so all we can do here is error out.
                                        // TODO: put this in find_image_conflict instead?
                                        panic!(
                                            "Command requires Preinitialized layout, but the \
                                            initial layout of the image is not Preinitialized"
                                        );
                                    }
                                    _ => {
                                        // Insert a layout transition.

                                        // A layout transition is a write, so if we perform one, we
                                        // need exclusive access.
                                        state.memory.exclusive = true; // TODO: is this correct?
                                        state.exclusive_any = true;
                                        barrier.new_layout = start_layout;
                                    }
                                }
                            }

                            self.pending_barrier.image_memory_barriers.push(barrier);
                        }
                        CommandBufferLevel::Secondary => {
                            state.initial_layout = start_layout;
                        }
                    }
                } else {
                    // This resource range was used before in this command buffer.

                    // If the command expects the image to be undefined, then we can't
                    // transition it, so use the current layout for both old and new layout.
                    let start_layout = if start_layout == ImageLayout::Undefined {
                        state.current_layout
                    } else {
                        start_layout
                    };

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
                        if state
                            .resource_uses
                            .iter()
                            .any(|resource_use| resource_use.command_index >= self.first_unflushed)
                            || state.current_layout != start_layout
                        {
                            unsafe {
                                // Flush the pending barrier.
                                self.inner.pipeline_barrier(&self.pending_barrier);
                                self.pending_barrier.clear();
                                self.barriers.push(self.first_unflushed); // Track inserted barriers

                                for command in &mut self.commands
                                    [self.first_unflushed..last_allowed_barrier_index]
                                {
                                    command.send(&mut self.inner);
                                }
                                self.first_unflushed = last_allowed_barrier_index;
                            }
                        }

                        // Modify the pipeline barrier to handle the collision.
                        self.pending_barrier
                            .image_memory_barriers
                            .push(ImageMemoryBarrier {
                                src_stages: state.memory.stages,
                                src_access: state.memory.access,
                                dst_stages: memory.stages,
                                dst_access: memory.access,
                                old_layout: state.current_layout,
                                new_layout: start_layout,
                                subresource_range: inner.image.range_to_subresources(range.clone()),
                                ..ImageMemoryBarrier::image(inner.image.clone())
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
                        state.memory.stages |= memory.stages;
                        state.memory.access |= memory.access;
                    }

                    state.resource_uses.push(ImageUse {
                        command_index: self.commands.len() - 1,
                        name: resource_name.clone(),
                    });
                }
            }
        }
    }

    /// Builds the command buffer and turns it into a `SyncCommandBuffer`.
    #[inline]
    pub fn build(mut self) -> Result<SyncCommandBuffer, OomError> {
        debug_assert!(self.latest_render_pass_enter.is_none() || self.pending_barrier.is_empty());

        // The commands that haven't been sent to the inner command buffer yet need to be sent.
        unsafe {
            self.inner.pipeline_barrier(&self.pending_barrier);
            self.pending_barrier.clear();
            let start = self.first_unflushed;
            self.barriers.push(start); // Track inserted barriers

            for command in &mut self.commands[start..] {
                command.send(&mut self.inner);
            }
        }

        // Transition images to their desired final layout.
        if self.level == CommandBufferLevel::Primary {
            unsafe {
                for (image, range_map) in self.images2.iter_mut() {
                    for (range, state) in range_map
                        .iter_mut()
                        .filter(|(_range, state)| state.final_layout != state.current_layout)
                    {
                        self.pending_barrier
                            .image_memory_barriers
                            .push(ImageMemoryBarrier {
                                src_stages: state.memory.stages,
                                src_access: state.memory.access,
                                dst_stages: PipelineStages::TOP_OF_PIPE,
                                dst_access: AccessFlags::empty(),
                                old_layout: state.current_layout,
                                new_layout: state.final_layout,
                                subresource_range: image.range_to_subresources(range.clone()),
                                ..ImageMemoryBarrier::image(image.clone())
                            });

                        state.exclusive_any = true;
                    }
                }

                self.inner.pipeline_barrier(&self.pending_barrier);
            }
        }

        let resource_usage = CommandBufferResourcesUsage {
            buffers: self
                .buffers2
                .into_iter()
                .map(|(buffer, ranges)| CommandBufferBufferUsage {
                    buffer,
                    ranges: ranges
                        .into_iter()
                        .filter(|(_range, state)| !state.resource_uses.is_empty())
                        .map(|(range, state)| {
                            let first_use = state.resource_uses.into_iter().next().unwrap();
                            (
                                range,
                                CommandBufferBufferRangeUsage {
                                    first_use: FirstResourceUse {
                                        command_index: first_use.command_index,
                                        command_name: self.commands[first_use.command_index].name(),
                                        description: first_use.name,
                                    },
                                    mutable: state.exclusive_any,
                                    final_stages: state.memory.stages,
                                    final_access: state.memory.access,
                                },
                            )
                        })
                        .collect(),
                })
                .collect(),
            images: self
                .images2
                .into_iter()
                .map(|(image, ranges)| CommandBufferImageUsage {
                    image,
                    ranges: ranges
                        .into_iter()
                        .filter(|(_range, state)| {
                            !state.resource_uses.is_empty()
                                || (self.level == CommandBufferLevel::Primary
                                    && state.current_layout != state.final_layout)
                        })
                        .map(|(range, mut state)| {
                            if self.level == CommandBufferLevel::Primary {
                                state.current_layout = state.final_layout;
                            }

                            let first_use = state.resource_uses.into_iter().next().unwrap();
                            (
                                range,
                                CommandBufferImageRangeUsage {
                                    first_use: FirstResourceUse {
                                        command_index: first_use.command_index,
                                        command_name: self.commands[first_use.command_index].name(),
                                        description: first_use.name,
                                    },
                                    mutable: state.exclusive_any,
                                    final_stages: state.memory.stages,
                                    final_access: state.memory.access,
                                    expected_layout: state.initial_layout,
                                    final_layout: state.current_layout,
                                },
                            )
                        })
                        .collect(),
                })
                .collect(),
        };

        let buffer_indices: HashMap<_, _> = resource_usage
            .buffers
            .iter()
            .enumerate()
            .map(|(index, usage)| (usage.buffer.clone(), index))
            .collect();
        let image_indices: HashMap<_, _> = resource_usage
            .images
            .iter()
            .enumerate()
            .map(|(index, usage)| (usage.image.clone(), index))
            .collect();

        Ok(SyncCommandBuffer {
            inner: self.inner.build()?,
            buffers: self.buffers,
            images: self.images,
            resources_usage: resource_usage,
            buffer_indices,
            image_indices,
            _commands: self.commands,
            _barriers: self.barriers,
        })
    }
}

unsafe impl DeviceOwned for SyncCommandBufferBuilder {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl Debug for SyncCommandBufferBuilder {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        Debug::fmt(&self.inner, f)
    }
}

/// Error returned if the builder detects that there's an unsolvable conflict.
#[derive(Debug, Clone)]
pub enum SyncCommandBufferBuilderError {
    /// Unsolvable conflict.
    Conflict {
        command_param: Cow<'static, str>,
        previous_command_name: &'static str,
        previous_command_offset: usize,
        previous_command_param: Cow<'static, str>,
    },

    ExecError(CommandBufferExecError),
}

impl Error for SyncCommandBufferBuilderError {}

impl Display for SyncCommandBufferBuilderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            SyncCommandBufferBuilderError::Conflict { .. } => write!(f, "unsolvable conflict"),
            SyncCommandBufferBuilderError::ExecError(err) => Display::fmt(err, f),
        }
    }
}

impl From<CommandBufferExecError> for SyncCommandBufferBuilderError {
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

    // The layout that the image range must have when this command buffer is executed.
    // Can be `Undefined` if we don't care.
    initial_layout: ImageLayout,

    // Current layout at this stage of the building.
    current_layout: ImageLayout,

    // The layout that the image range will have at the end of the command buffer.
    // This is only used for primary command buffers.
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
    pub(in crate::command_buffer) depth_bounds: Option<RangeInclusive<f32>>,
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

                let invalidate_from = if state.pipeline_layout.handle() == pipeline_layout.handle()
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
    #[inline]
    pub fn resources(&self) -> &DescriptorSetResources {
        match self {
            Self::Set(set) => set.as_ref().0.resources(),
            Self::Push(resources) => resources,
        }
    }
}

/// Allows you to retrieve the current state of a command buffer builder.
#[derive(Clone, Copy)]
pub struct CommandBufferBuilderState<'a> {
    current_state: &'a CurrentState,
}

impl<'a> CommandBufferBuilderState<'a> {
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
    pub fn depth_bounds(&self) -> Option<RangeInclusive<f32>> {
        self.current_state.depth_bounds.clone()
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

    /// Returns the current scissor for a given viewport slot, or `None` if nothing has been set
    /// yet.
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

    /// Returns the current viewport for a given viewport slot, or `None` if nothing has been set
    /// yet.
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
