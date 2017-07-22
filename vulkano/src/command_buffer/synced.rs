// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use fnv::FnvHashMap;
use smallvec::SmallVec;
use std::any::Any;
use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

use OomError;
use buffer::BufferAccess;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferExecError;
use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolAlloc;
use command_buffer::pool::CommandPoolBuilderAlloc;
use command_buffer::sys::Flags;
use command_buffer::sys::Kind;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use command_buffer::sys::UnsafeCommandBufferBuilderBindVertexBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilderBufferImageCopy;
use command_buffer::sys::UnsafeCommandBufferBuilderColorImageClear;
use command_buffer::sys::UnsafeCommandBufferBuilderExecuteCommands;
use command_buffer::sys::UnsafeCommandBufferBuilderImageBlit;
use command_buffer::sys::UnsafeCommandBufferBuilderPipelineBarrier;
use descriptor::descriptor::DescriptorDescTy;
use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use format::ClearValue;
use framebuffer::FramebufferAbstract;
use framebuffer::RenderPassAbstract;
use framebuffer::SubpassContents;
use image::ImageAccess;
use image::ImageLayout;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use pipeline::input_assembly::IndexType;
use pipeline::viewport::Scissor;
use pipeline::viewport::Viewport;
use sampler::Filter;
use sync::AccessCheckError;
use sync::AccessError;
use sync::AccessFlagBits;
use sync::Event;
use sync::GpuFuture;
use sync::PipelineStages;

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
/// the commands except for synchronization purposes. The builder may panic if you pass unvalid
/// commands.
///
/// The `P` generic is the same as `UnsafeCommandBufferBuilder`.
pub struct SyncCommandBufferBuilder<P> {
    // The actual Vulkan command buffer builder.
    inner: UnsafeCommandBufferBuilder<P>,

    // Stores the current state of all resources that are in use by the command buffer.
    resources: FnvHashMap<BuilderKey<P>, ResourceState>,

    // Prototype for the pipeline barrier that must be submitted before flushing the commands
    // in `commands`.
    pending_barrier: UnsafeCommandBufferBuilderPipelineBarrier,

    // Stores all the commands that were submitted or are going to be submitted to the inner
    // builder. A copy of this `Arc` is stored in each `BuilderKey`.
    commands: Arc<Mutex<Commands<P>>>,

    // True if we're a secondary command buffer.
    is_secondary: bool,
}

impl<P> fmt::Debug for SyncCommandBufferBuilder<P> {
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
}

impl error::Error for SyncCommandBufferBuilderError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            SyncCommandBufferBuilderError::Conflict { .. } => {
                "unsolvable conflict"
            },
        }
    }
}

impl fmt::Display for SyncCommandBufferBuilderError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

// List of commands of a `SyncCommandBufferBuilder`.
struct Commands<P> {
    // Only the commands before `first_unflushed` have already been sent to the inner
    // `UnsafeCommandBufferBuilder`.
    first_unflushed: usize,

    // If we're currently inside a render pass, contains the index of the `CmdBeginRenderPass`
    // command.
    latest_render_pass_enter: Option<usize>,

    // The actual list.
    commands: Vec<Box<Command<P> + Send + Sync>>,
}

// A single command within the list of commands.
trait Command<P> {
    // Returns a user-friendly name for the command for error reporting purposes.
    fn name(&self) -> &'static str;

    // Sends the command to the `UnsafeCommandBufferBuilder`. Calling this method twice on the same
    // object may lead to a panic.
    unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>);

    // Turns this command into a `FinalCommand`.
    fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync>;

    // Gives access to the `num`th buffer used by the command.
    fn buffer(&self, num: usize) -> &BufferAccess {
        panic!()
    }
    // Gives access to the `num`th image used by the command.
    fn image(&self, num: usize) -> &ImageAccess {
        panic!()
    }

    // Returns a user-friendly name for the `num`th buffer used by the command, for error
    // reporting purposes.
    fn buffer_name(&self, num: usize) -> Cow<'static, str> {
        panic!()
    }
    // Returns a user-friendly name for the `num`th image used by the command, for error
    // reporting purposes.
    fn image_name(&self, num: usize) -> Cow<'static, str> {
        panic!()
    }
}

// Key that identifies a resource. Implements `PartialEq`, `Eq` and `Hash` so that two resources
// that conflict with each other compare equal.
//
// This works by holding an Arc to the list of commands and the index of the command that holds
// the resource.
struct BuilderKey<P> {
    // Same `Arc` as the `SyncCommandBufferBuilder`.
    commands: Arc<Mutex<Commands<P>>>,
    // Index of the command that holds the resource within `commands`.
    command_id: usize,
    // Type of the resource.
    resource_ty: KeyTy,
    // Index of the resource within the command.
    resource_index: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum KeyTy {
    Buffer,
    Image,
}

impl<P> BuilderKey<P> {
    // Turns this key used by the builder into a key used by the final command buffer.
    // Called when the command buffer is being built.
    fn into_cb_key(self, final_commands: Arc<Mutex<Vec<Box<FinalCommand + Send + Sync>>>>)
                   -> CbKey<'static> {
        CbKey::Command {
            commands: final_commands,
            command_id: self.command_id,
            resource_ty: self.resource_ty,
            resource_index: self.resource_index,
        }
    }

    #[inline]
    fn conflicts_buffer_all(&self, commands_lock: &Commands<P>, buf: &BufferAccess) -> bool {
        // TODO: put the conflicts_* methods directly on the Command trait to avoid an indirect call?
        match self.resource_ty {
            KeyTy::Buffer => {
                let c = &commands_lock.commands[self.command_id];
                c.buffer(self.resource_index).conflicts_buffer_all(buf)
            },
            KeyTy::Image => {
                let c = &commands_lock.commands[self.command_id];
                c.image(self.resource_index).conflicts_buffer_all(buf)
            },
        }
    }

    #[inline]
    fn conflicts_image_all(&self, commands_lock: &Commands<P>, img: &ImageAccess) -> bool {
        // TODO: put the conflicts_* methods directly on the Command trait to avoid an indirect call?
        match self.resource_ty {
            KeyTy::Buffer => {
                let c = &commands_lock.commands[self.command_id];
                c.buffer(self.resource_index).conflicts_image_all(img)
            },
            KeyTy::Image => {
                let c = &commands_lock.commands[self.command_id];
                c.image(self.resource_index).conflicts_image_all(img)
            },
        }
    }
}

impl<P> PartialEq for BuilderKey<P> {
    #[inline]
    fn eq(&self, other: &BuilderKey<P>) -> bool {
        debug_assert!(Arc::ptr_eq(&self.commands, &other.commands));
        let commands_lock = self.commands.lock().unwrap();

        match other.resource_ty {
            KeyTy::Buffer => {
                let c = &commands_lock.commands[other.command_id];
                self.conflicts_buffer_all(&commands_lock, c.buffer(other.resource_index))
            },
            KeyTy::Image => {
                let c = &commands_lock.commands[other.command_id];
                self.conflicts_image_all(&commands_lock, c.image(other.resource_index))
            },
        }
    }
}

impl<P> Eq for BuilderKey<P> {
}

impl<P> Hash for BuilderKey<P> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let commands_lock = self.commands.lock().unwrap();

        match self.resource_ty {
            KeyTy::Buffer => {
                let c = &commands_lock.commands[self.command_id];
                c.buffer(self.resource_index).conflict_key_all()
            },
            KeyTy::Image => {
                let c = &commands_lock.commands[self.command_id];
                c.image(self.resource_index).conflict_key_all()
            },
        }.hash(state)
    }
}

// Current state of a resource during the building of the command buffer.
#[derive(Debug, Clone)]
struct ResourceState {
    // Stages of the command that last used this resource.
    stages: PipelineStages,
    // Access for the command that last used this resource.
    access: AccessFlagBits,

    // True if the resource was used in exclusive mode at any point during the building of the
    // command buffer.
    exclusive_any: bool,

    // True if the last command that used this resource used it in exclusive mode.
    exclusive: bool,

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
            final_stages: self.stages,
            final_access: self.access,
            exclusive: self.exclusive_any,
            initial_layout: self.initial_layout,
            final_layout: self.current_layout,
        }
    }
}

impl<P> SyncCommandBufferBuilder<P> {
    /// Builds a new `SyncCommandBufferBuilder`. The parameters are the same as the
    /// `UnsafeCommandBufferBuilder::new` function.
    ///
    /// # Safety
    ///
    /// See `UnsafeCommandBufferBuilder::new()` and `SyncCommandBufferBuilder`.
    pub unsafe fn new<Pool, R, F, A>(pool: &Pool, kind: Kind<R, F>, flags: Flags)
                                     -> Result<SyncCommandBufferBuilder<P>, OomError>
        where Pool: CommandPool<Builder = P, Alloc = A>,
              P: CommandPoolBuilderAlloc<Alloc = A>,
              A: CommandPoolAlloc,
              R: RenderPassAbstract,
              F: FramebufferAbstract
    {
        let (is_secondary, inside_render_pass) = match kind {
            Kind::Primary => (false, false),
            Kind::Secondary { ref render_pass, .. } => (true, render_pass.is_some()),
        };

        let cmd = UnsafeCommandBufferBuilder::new(pool, kind, flags)?;
        Ok(SyncCommandBufferBuilder::from_unsafe_cmd(cmd, is_secondary, inside_render_pass))
    }

    /// Builds a `SyncCommandBufferBuilder` from an existing `UnsafeCommandBufferBuilder`.
    ///
    /// # Safety
    ///
    /// See `UnsafeCommandBufferBuilder::new()` and `SyncCommandBufferBuilder`.
    ///
    /// In addition to this, the `UnsafeCommandBufferBuilder` should be empty. If it isn't, then
    /// you must take into account the fact that the `SyncCommandBufferBuilder` won't be aware of
    /// any existing resource usage.
    #[inline]
    pub unsafe fn from_unsafe_cmd(cmd: UnsafeCommandBufferBuilder<P>, is_secondary: bool,
                                  inside_render_pass: bool) -> SyncCommandBufferBuilder<P> {
        let latest_render_pass_enter = if inside_render_pass {
            Some(0)
        } else {
            None
        };
                                
        SyncCommandBufferBuilder {
            inner: cmd,
            resources: FnvHashMap::default(),
            pending_barrier: UnsafeCommandBufferBuilderPipelineBarrier::new(),
            commands: Arc::new(Mutex::new(Commands {
                                              first_unflushed: 0,
                                              latest_render_pass_enter,
                                              commands: Vec::new(),
                                          })),
            is_secondary,
        }
    }

    // After a command is added to the list of pending commands, this function must be called for
    // each resource used by the command that has just been added.
    // The function will take care of handling the pipeline barrier or flushing.
    fn prev_cmd_resource(&mut self, resource_ty: KeyTy, resource_index: usize, exclusive: bool,
                         stages: PipelineStages, access: AccessFlagBits,
                         start_layout: ImageLayout, end_layout: ImageLayout)
                         -> Result<(), SyncCommandBufferBuilderError> {
        debug_assert!(exclusive || start_layout == end_layout);
        debug_assert!(access.is_compatible_with(&stages));
        debug_assert!(resource_ty != KeyTy::Image || end_layout != ImageLayout::Undefined);
        debug_assert!(resource_ty != KeyTy::Buffer || start_layout == ImageLayout::Undefined);
        debug_assert!(resource_ty != KeyTy::Buffer || end_layout == ImageLayout::Undefined);
        debug_assert_ne!(end_layout, ImageLayout::Preinitialized);

        let (first_unflushed, latest_command_id) = {
            let commands_lock = self.commands.lock().unwrap();
            debug_assert!(commands_lock.commands.len() >= 1);
            (commands_lock.first_unflushed, commands_lock.commands.len() - 1)
        };

        let key = BuilderKey {
            commands: self.commands.clone(),
            command_id: latest_command_id,
            resource_ty,
            resource_index,
        };

        match self.resources.entry(key) {
            Entry::Occupied(entry) => {
                let collision_command_id = entry.key().command_id;
                debug_assert!(collision_command_id <= latest_command_id);

                let entry_key_resource_index = entry.key().resource_index;
                let entry_key_resource_ty = entry.key().resource_ty;
                let mut entry = entry.into_mut();

                // Find out if we have a collision with the pending commands.
                if exclusive || entry.exclusive || entry.current_layout != start_layout {
                    // Collision found.

                    // We now want to modify the current pipeline barrier in order to include the
                    // transition. But since the pipeline barrier is going to be submitted before
                    // the flushed commands, it would be a mistake if the command we transition
                    // from hasn't been flushed yet.
                    if collision_command_id >= first_unflushed {
                        // Flush.
                        unsafe {
                            self.inner.pipeline_barrier(&self.pending_barrier);
                            self.pending_barrier = UnsafeCommandBufferBuilderPipelineBarrier::new();
                            {
                                let mut commands_lock = self.commands.lock().unwrap();
                                let start = commands_lock.first_unflushed;
                                let end = if let Some(rp_enter) = commands_lock.latest_render_pass_enter {
                                    rp_enter
                                } else {
                                    latest_command_id
                                };
                                if collision_command_id >= end {
                                    return Err(SyncCommandBufferBuilderError::Conflict {
                                        command1_name: commands_lock.commands[collision_command_id].name(),
                                        command1_param: match entry_key_resource_ty {
                                            KeyTy::Buffer => commands_lock.commands[collision_command_id].buffer_name(entry_key_resource_index),
                                            KeyTy::Image => commands_lock.commands[collision_command_id].image_name(entry_key_resource_index),
                                        },
                                        command1_offset: collision_command_id,

                                        command2_name: commands_lock.commands[latest_command_id].name(),
                                        command2_param: match resource_ty {
                                            KeyTy::Buffer => commands_lock.commands[latest_command_id].buffer_name(resource_index),
                                            KeyTy::Image => commands_lock.commands[latest_command_id].image_name(resource_index),
                                        },
                                        command2_offset: latest_command_id,
                                    });
                                }
                                for command in &mut commands_lock.commands[start .. end] {
                                    command.send(&mut self.inner);
                                }
                                commands_lock.first_unflushed = end;
                            }
                        }
                    }

                    // Modify the pipeline barrier to include the transition.
                    unsafe {
                        let commands_lock = self.commands.lock().unwrap();
                        match resource_ty {
                            KeyTy::Buffer => {
                                let buf = commands_lock.commands[latest_command_id]
                                    .buffer(resource_index);

                                let b = &mut self.pending_barrier;
                                b.add_buffer_memory_barrier(buf,
                                                            entry.stages,
                                                            entry.access,
                                                            stages,
                                                            access,
                                                            true,
                                                            None,
                                                            0,
                                                            buf.size());
                            },

                            KeyTy::Image => {
                                let img = commands_lock.commands[latest_command_id]
                                    .image(resource_index);

                                let b = &mut self.pending_barrier;
                                b.add_image_memory_barrier(img,
                                                           0 .. img.mipmap_levels(),
                                                           0 .. img.dimensions().array_layers(),
                                                           entry.stages,
                                                           entry.access,
                                                           stages,
                                                           access,
                                                           true,
                                                           None,
                                                           entry.current_layout,
                                                           start_layout);
                            },
                        };
                    }

                    // Update state.
                    entry.stages = stages;
                    entry.access = access;
                    entry.exclusive_any = true;
                    entry.exclusive = exclusive;
                    if exclusive || end_layout != ImageLayout::Undefined {
                        // Only modify the layout in case of a write, because buffer operations
                        // pass `Undefined` for the layout. While a buffer write *must* set the
                        // layout to `Undefined`, a buffer read must not touch it.
                        entry.current_layout = end_layout;
                    }

                } else {
                    entry.stages = entry.stages | stages;
                    entry.access = entry.access | access;
                }
            },

            Entry::Vacant(entry) => {
                let mut actually_exclusive = exclusive;

                // Handle the case when the initial layout requirement of the image is different
                // from the first layout usage.
                if !self.is_secondary && resource_ty == KeyTy::Image &&
                    start_layout != ImageLayout::Undefined &&
                    start_layout != ImageLayout::Preinitialized
                {
                    let commands_lock = self.commands.lock().unwrap();
                    let img = commands_lock.commands[latest_command_id].image(resource_index);

                    if img.initial_layout_requirement() != start_layout {
                        actually_exclusive = true;

                        unsafe {
                            let b = &mut self.pending_barrier;
                            b.add_image_memory_barrier(img,
                                                       0 .. img.mipmap_levels(),
                                                       0 .. img.dimensions().array_layers(),
                                                       PipelineStages {
                                                           bottom_of_pipe: true,
                                                           ..PipelineStages::none()
                                                       }, // TODO:?
                                                       AccessFlagBits::none(), // TODO: ?
                                                       stages,
                                                       access,
                                                       true,
                                                       None,
                                                       img.initial_layout_requirement(),
                                                       start_layout);
                        }
                    }
                }

                entry.insert(ResourceState {
                    stages: stages,
                    access: access,
                    exclusive_any: actually_exclusive,
                    exclusive: actually_exclusive,
                    initial_layout: start_layout,
                    current_layout: end_layout,     // TODO: what if we reach the end with Undefined? that's not correct?
                });
            },
        }

        Ok(())
    }

    /// Builds the command buffer.
    #[inline]
    pub fn build(mut self) -> Result<SyncCommandBuffer<P::Alloc>, OomError>
        where P: CommandPoolBuilderAlloc
    {
        let mut commands_lock = self.commands.lock().unwrap();
        debug_assert!(commands_lock.latest_render_pass_enter.is_none() ||
                      self.pending_barrier.is_empty());

        // Flush the commands that haven't been flushed yet.
        unsafe {
            self.inner.pipeline_barrier(&self.pending_barrier);
            let f = commands_lock.first_unflushed;
            for command in &mut commands_lock.commands[f ..] {
                command.send(&mut self.inner);
            }
        }

        // Transition images to their desired final layout.
        if !self.is_secondary {
            unsafe {
                let mut barrier = UnsafeCommandBufferBuilderPipelineBarrier::new();

                for (key, mut state) in &mut self.resources {
                    if key.resource_ty != KeyTy::Image {
                        continue;
                    }

                    let img = commands_lock.commands[key.command_id].image(key.resource_index);
                    let requested_layout = img.final_layout_requirement();
                    if requested_layout == state.current_layout {
                        continue;
                    }

                    state.exclusive_any = true;
                    state.current_layout = requested_layout;
                    barrier.add_image_memory_barrier(img,
                                                    0 .. img.mipmap_levels(),
                                                    0 .. img.dimensions().array_layers(),
                                                    state.stages,
                                                    state.access,
                                                    PipelineStages {
                                                        bottom_of_pipe: true,
                                                        ..PipelineStages::none()
                                                    }, // TODO:?
                                                    AccessFlagBits::none(),
                                                    true,
                                                    None, // TODO: access?
                                                    state.current_layout,
                                                    requested_layout);
                }

                self.inner.pipeline_barrier(&barrier);
            }
        }

        // Fill the `commands` list.
        let final_commands = {
            let mut final_commands = Vec::new();
            for command in commands_lock.commands.drain(..) {
                final_commands.push(command.into_final_command());
            }
            Arc::new(Mutex::new(final_commands))
        };

        // Build the final resources states.
        let final_resources_states: FnvHashMap<_, _> = {
            self.resources
                .into_iter()
                .map(|(resource, state)| {
                         (resource.into_cb_key(final_commands.clone()), state.finalize())
                     })
                .collect()
        };

        Ok(SyncCommandBuffer {
               inner: self.inner.build()?,
               resources: final_resources_states,
               commands: final_commands,
           })
    }

    /// Calls `vkBeginRenderPass` on the builder.
    // TODO: it shouldn't be possible to get an error if the framebuffer checked conflicts already
    // TODO: after begin_render_pass has been called, flushing should be forbidden and an error
    //       returned if conflict
    #[inline]
    pub unsafe fn begin_render_pass<F, I>(&mut self, framebuffer: F,
                                          subpass_contents: SubpassContents, clear_values: I)
                                          -> Result<(), SyncCommandBufferBuilderError>
        where F: FramebufferAbstract + Send + Sync + 'static,
              I: Iterator<Item = ClearValue> + Send + Sync + 'static
    {
        struct Cmd<F, I> {
            framebuffer: F,
            subpass_contents: SubpassContents,
            clear_values: Option<I>,
        }

        impl<P, F, I> Command<P> for Cmd<F, I>
            where F: FramebufferAbstract + Send + Sync + 'static,
                  I: Iterator<Item = ClearValue>
        {
            fn name(&self) -> &'static str {
                "vkCmdBeginRenderPass"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.begin_render_pass(&self.framebuffer,
                                      self.subpass_contents,
                                      self.clear_values.take().unwrap());
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<F>(F);
                impl<F> FinalCommand for Fin<F>
                    where F: FramebufferAbstract + Send + Sync + 'static
                {
                    fn image(&self, num: usize) -> &ImageAccess {
                        self.0.attached_image_view(num).unwrap().parent()
                    }
                }
                Box::new(Fin(self.framebuffer))
            }

            fn image(&self, num: usize) -> &ImageAccess {
                self.framebuffer.attached_image_view(num).unwrap().parent()
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                format!("attachment {}", num).into()
            }
        }

        let atch_desc = (0 .. framebuffer.num_attachments())
            .map(|atch| framebuffer.attachment_desc(atch).unwrap())
            .collect::<Vec<_>>();

        // FIXME: this is bad because dropping the command buffer doesn't drop the
        //        attachments of the framebuffer, meaning that they will stay locked
        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 framebuffer,
                                                                 subpass_contents,
                                                                 clear_values: Some(clear_values),
                                                             }));

        for (atch, desc) in atch_desc.into_iter().enumerate() {
            self.prev_cmd_resource(KeyTy::Image, atch, true,        // TODO: suboptimal ; note: remember to always pass true if desc.initial_layout != desc.final_layout
                                   PipelineStages {
                                       all_commands: true,
                                       .. PipelineStages::none()
                                   },       // TODO: wrong!
                                   AccessFlagBits {
                                       input_attachment_read: true,
                                       color_attachment_read: true,
                                       color_attachment_write: true,
                                       depth_stencil_attachment_read: true,
                                       depth_stencil_attachment_write: true,
                                       .. AccessFlagBits::none()
                                   },       // TODO: suboptimal
                                   desc.initial_layout, desc.final_layout)?;
        }

        {
            let mut cmd_lock = self.commands.lock().unwrap();
            cmd_lock.latest_render_pass_enter = Some(cmd_lock.commands.len() - 1);
        }

        Ok(())
    }

    /// Calls `vkCmdBindIndexBuffer` on the builder.
    #[inline]
    pub unsafe fn bind_index_buffer<B>(&mut self, buffer: B, index_ty: IndexType)
                                       -> Result<(), SyncCommandBufferBuilderError>
        where B: BufferAccess + Send + Sync + 'static
    {
        struct Cmd<B> {
            buffer: B,
            index_ty: IndexType,
        }

        impl<P, B> Command<P> for Cmd<B>
            where B: BufferAccess + Send + Sync + 'static
        {
            fn name(&self) -> &'static str {
                "vkCmdBindIndexBuffer"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.bind_index_buffer(&self.buffer, self.index_ty);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                    where B: BufferAccess + Send + Sync + 'static
                {
                    fn buffer(&self, num: usize) -> &BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "index buffer".into()
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { buffer, index_ty }));
        self.prev_cmd_resource(KeyTy::Buffer,
                               0,
                               false,
                               PipelineStages {
                                   vertex_input: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   index_read: true,
                                   ..AccessFlagBits::none()
                               },
                               ImageLayout::Undefined,
                               ImageLayout::Undefined)?;
        Ok(())
    }

    /// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_graphics<Gp>(&mut self, pipeline: Gp)
        where Gp: GraphicsPipelineAbstract + Send + Sync + 'static
    {
        struct Cmd<Gp> {
            pipeline: Gp,
        }

        impl<P, Gp> Command<P> for Cmd<Gp>
            where Gp: GraphicsPipelineAbstract + Send + Sync + 'static
        {
            fn name(&self) -> &'static str {
                "vkCmdBindPipeline"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.bind_pipeline_graphics(&self.pipeline);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<Gp>(Gp);
                impl<Gp> FinalCommand for Fin<Gp>
                    where Gp: Send + Sync + 'static
                {
                }
                Box::new(Fin(self.pipeline))
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { pipeline }));
    }

    /// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_compute<Cp>(&mut self, pipeline: Cp)
        where Cp: ComputePipelineAbstract + Send + Sync + 'static
    {
        struct Cmd<Gp> {
            pipeline: Gp,
        }

        impl<P, Gp> Command<P> for Cmd<Gp>
            where Gp: ComputePipelineAbstract + Send + Sync + 'static
        {
            fn name(&self) -> &'static str {
                "vkCmdBindPipeline"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.bind_pipeline_compute(&self.pipeline);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<Cp>(Cp);
                impl<Cp> FinalCommand for Fin<Cp>
                    where Cp: Send + Sync + 'static
                {
                }
                Box::new(Fin(self.pipeline))
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { pipeline }));
    }

    /// Starts the process of binding descriptor sets. Returns an intermediate struct which can be
    /// used to add the sets.
    #[inline]
    pub fn bind_descriptor_sets(&mut self) -> SyncCommandBufferBuilderBindDescriptorSets<P> {
        SyncCommandBufferBuilderBindDescriptorSets {
            builder: self,
            inner: SmallVec::new(),
        }
    }

    /// Starts the process of binding vertex buffers. Returns an intermediate struct which can be
    /// used to add the buffers.
    #[inline]
    pub fn bind_vertex_buffers(&mut self) -> SyncCommandBufferBuilderBindVertexBuffer<P> {
        SyncCommandBufferBuilderBindVertexBuffer {
            builder: self,
            inner: UnsafeCommandBufferBuilderBindVertexBuffer::new(),
            buffers: Vec::new(),
        }
    }

    /// Calls `vkCmdBlitImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn blit_image<S, D, R>(&mut self, source: S, source_layout: ImageLayout,
                                      destination: D, destination_layout: ImageLayout, regions: R,
                                      filter: Filter)
                                      -> Result<(), SyncCommandBufferBuilderError>
        where S: ImageAccess + Send + Sync + 'static,
              D: ImageAccess + Send + Sync + 'static,
              R: Iterator<Item = UnsafeCommandBufferBuilderImageBlit> + Send + Sync + 'static
    {
        struct Cmd<S, D, R> {
            source: Option<S>,
            source_layout: ImageLayout,
            destination: Option<D>,
            destination_layout: ImageLayout,
            regions: Option<R>,
            filter: Filter,
        }

        impl<P, S, D, R> Command<P> for Cmd<S, D, R>
            where S: ImageAccess + Send + Sync + 'static,
                  D: ImageAccess + Send + Sync + 'static,
                  R: Iterator<Item = UnsafeCommandBufferBuilderImageBlit>
        {
            fn name(&self) -> &'static str {
                "vkCmdBlitImage"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.blit_image(self.source.as_ref().unwrap(), self.source_layout,
                               self.destination.as_ref().unwrap(), self.destination_layout,
                               self.regions.take().unwrap(), self.filter);
            }

            fn into_final_command(mut self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<S, D>(S, D);
                impl<S, D> FinalCommand for Fin<S, D>
                    where S: ImageAccess + Send + Sync + 'static,
                          D: ImageAccess + Send + Sync + 'static
                {
                    fn image(&self, num: usize) -> &ImageAccess {
                        if num == 0 {
                            &self.0
                        } else if num == 1 {
                            &self.1
                        } else {
                            panic!()
                        }
                    }
                }

                // Note: borrow checker somehow doesn't accept `self.source` and `self.destination`
                // without using an Option.
                Box::new(Fin(self.source.take().unwrap(),
                             self.destination.take().unwrap()))
            }

            fn image(&self, num: usize) -> &ImageAccess {
                if num == 0 {
                    self.source.as_ref().unwrap()
                } else if num == 1 {
                    self.destination.as_ref().unwrap()
                } else {
                    panic!()
                }
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                if num == 0 {
                    "source".into()
                } else if num == 1 {
                    "destination".into()
                } else {
                    panic!()
                }
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 source: Some(source),
                                                                 source_layout,
                                                                 destination: Some(destination),
                                                                 destination_layout,
                                                                 regions: Some(regions),
                                                                 filter,
                                                             }));
        self.prev_cmd_resource(KeyTy::Image,
                               0,
                               false,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_read: true,
                                   ..AccessFlagBits::none()
                               },
                               source_layout,
                               source_layout)?;
        self.prev_cmd_resource(KeyTy::Image,
                               1,
                               true,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_write: true,
                                   ..AccessFlagBits::none()
                               },
                               destination_layout,
                               destination_layout)?;
        Ok(())
    }

    /// Calls `vkCmdClearColorImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    pub unsafe fn clear_color_image<I, R>(&mut self, image: I, layout: ImageLayout,
                                          color: ClearValue, regions: R)
                                          -> Result<(), SyncCommandBufferBuilderError>
        where I: ImageAccess + Send + Sync + 'static,
              R: Iterator<Item = UnsafeCommandBufferBuilderColorImageClear> + Send + Sync + 'static
    {
        struct Cmd<I, R> {
            image: Option<I>,
            layout: ImageLayout,
            color: ClearValue,
            regions: Option<R>,
        }

        impl<P, I, R> Command<P> for Cmd<I, R>
            where I: ImageAccess + Send + Sync + 'static,
                  R: Iterator<Item = UnsafeCommandBufferBuilderColorImageClear> + Send + Sync + 'static
        {
            fn name(&self) -> &'static str {
                "vkCmdClearColorImage"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.clear_color_image(self.image.as_ref().unwrap(), self.layout, self.color,
                                      self.regions.take().unwrap());
            }

            fn into_final_command(mut self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<I>(I);
                impl<I> FinalCommand for Fin<I>
                    where I: ImageAccess + Send + Sync + 'static,
                {
                    fn image(&self, num: usize) -> &ImageAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                }

                // Note: borrow checker somehow doesn't accept `self.image` without using an Option.
                Box::new(Fin(self.image.take().unwrap()))
            }

            fn image(&self, num: usize) -> &ImageAccess {
                assert_eq!(num, 0);
                self.image.as_ref().unwrap()
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "target".into()
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 image: Some(image),
                                                                 layout,
                                                                 color,
                                                                 regions: Some(regions),
                                                             }));
        self.prev_cmd_resource(KeyTy::Image,
                               0,
                               true,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_write: true,
                                   ..AccessFlagBits::none()
                               },
                               layout,
                               layout)?;
        Ok(())
    }

    /// Calls `vkCmdCopyBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer<S, D, R>(&mut self, source: S, destination: D, regions: R)
                                       -> Result<(), SyncCommandBufferBuilderError>
        where S: BufferAccess + Send + Sync + 'static,
              D: BufferAccess + Send + Sync + 'static,
              R: Iterator<Item = (usize, usize, usize)> + Send + Sync + 'static
    {
        struct Cmd<S, D, R> {
            source: Option<S>,
            destination: Option<D>,
            regions: Option<R>,
        }

        impl<P, S, D, R> Command<P> for Cmd<S, D, R>
            where S: BufferAccess + Send + Sync + 'static,
                  D: BufferAccess + Send + Sync + 'static,
                  R: Iterator<Item = (usize, usize, usize)>
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyBuffer"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.copy_buffer(self.source.as_ref().unwrap(),
                                self.destination.as_ref().unwrap(),
                                self.regions.take().unwrap());
            }

            fn into_final_command(mut self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<S, D>(S, D);
                impl<S, D> FinalCommand for Fin<S, D>
                    where S: BufferAccess + Send + Sync + 'static,
                          D: BufferAccess + Send + Sync + 'static
                {
                    fn buffer(&self, num: usize) -> &BufferAccess {
                        match num {
                            0 => &self.0,
                            1 => &self.1,
                            _ => panic!(),
                        }
                    }
                }
                // Note: borrow checker somehow doesn't accept `self.source` and `self.destination`
                // without using an Option.
                Box::new(Fin(self.source.take().unwrap(),
                             self.destination.take().unwrap()))
            }

            fn buffer(&self, num: usize) -> &BufferAccess {
                match num {
                    0 => self.source.as_ref().unwrap(),
                    1 => self.destination.as_ref().unwrap(),
                    _ => panic!(),
                }
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                match num {
                    0 => "source".into(),
                    1 => "destination".into(),
                    _ => panic!(),
                }
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 source: Some(source),
                                                                 destination: Some(destination),
                                                                 regions: Some(regions),
                                                             }));
        self.prev_cmd_resource(KeyTy::Buffer,
                               0,
                               false,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_read: true,
                                   ..AccessFlagBits::none()
                               },
                               ImageLayout::Undefined,
                               ImageLayout::Undefined)?;
        self.prev_cmd_resource(KeyTy::Buffer,
                               1,
                               true,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_write: true,
                                   ..AccessFlagBits::none()
                               },
                               ImageLayout::Undefined,
                               ImageLayout::Undefined)?;
        Ok(())
    }

    /// Calls `vkCmdCopyBufferToImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer_to_image<S, D, R>(&mut self, source: S, destination: D,
                                                destination_layout: ImageLayout, regions: R)
                                                -> Result<(), SyncCommandBufferBuilderError>
        where S: BufferAccess + Send + Sync + 'static,
              D: ImageAccess + Send + Sync + 'static,
              R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy> + Send + Sync + 'static
    {
        struct Cmd<S, D, R> {
            source: Option<S>,
            destination: Option<D>,
            destination_layout: ImageLayout,
            regions: Option<R>,
        }

        impl<P, S, D, R> Command<P> for Cmd<S, D, R>
            where S: BufferAccess + Send + Sync + 'static,
                  D: ImageAccess + Send + Sync + 'static,
                  R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyBufferToImage"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.copy_buffer_to_image(self.source.as_ref().unwrap(),
                                         self.destination.as_ref().unwrap(),
                                         self.destination_layout,
                                         self.regions.take().unwrap());
            }

            fn into_final_command(mut self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<S, D>(S, D);
                impl<S, D> FinalCommand for Fin<S, D>
                    where S: BufferAccess + Send + Sync + 'static,
                          D: ImageAccess + Send + Sync + 'static
                {
                    fn buffer(&self, num: usize) -> &BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }

                    fn image(&self, num: usize) -> &ImageAccess {
                        assert_eq!(num, 0);
                        &self.1
                    }
                }

                // Note: borrow checker somehow doesn't accept `self.source` and `self.destination`
                // without using an Option.
                Box::new(Fin(self.source.take().unwrap(),
                             self.destination.take().unwrap()))
            }

            fn buffer(&self, num: usize) -> &BufferAccess {
                assert_eq!(num, 0);
                self.source.as_ref().unwrap()
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "source".into()
            }

            fn image(&self, num: usize) -> &ImageAccess {
                assert_eq!(num, 0);
                self.destination.as_ref().unwrap()
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "destination".into()
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 source: Some(source),
                                                                 destination: Some(destination),
            destination_layout: destination_layout,
                                                                 regions: Some(regions),
                                                             }));
        self.prev_cmd_resource(KeyTy::Buffer,
                               0,
                               false,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_read: true,
                                   ..AccessFlagBits::none()
                               },
                               ImageLayout::Undefined,
                               ImageLayout::Undefined)?;
        self.prev_cmd_resource(KeyTy::Image,
                               0,
                               true,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_write: true,
                                   ..AccessFlagBits::none()
                               },
                               destination_layout,
                               destination_layout)?;
        Ok(())
    }

    /// Calls `vkCmdCopyImageToBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image_to_buffer<S, D, R>(&mut self, source: S, source_layout: ImageLayout,
                                                destination: D, regions: R)
                                                -> Result<(), SyncCommandBufferBuilderError>
        where S: ImageAccess + Send + Sync + 'static,
              D: BufferAccess + Send + Sync + 'static,
              R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy> + Send + Sync + 'static
    {
        struct Cmd<S, D, R> {
            source: Option<S>,
            source_layout: ImageLayout,
            destination: Option<D>,
            regions: Option<R>,
        }

        impl<P, S, D, R> Command<P> for Cmd<S, D, R>
            where S: ImageAccess + Send + Sync + 'static,
                  D: BufferAccess + Send + Sync + 'static,
                  R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyImageToBuffer"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.copy_image_to_buffer(self.source.as_ref().unwrap(),
                                         self.source_layout,
                                         self.destination.as_ref().unwrap(),
                                         self.regions.take().unwrap());
            }

            fn into_final_command(mut self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<S, D>(S, D);
                impl<S, D> FinalCommand for Fin<S, D>
                    where S: ImageAccess + Send + Sync + 'static,
                          D: BufferAccess + Send + Sync + 'static
                {
                    fn buffer(&self, num: usize) -> &BufferAccess {
                        assert_eq!(num, 0);
                        &self.1
                    }

                    fn image(&self, num: usize) -> &ImageAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                }

                // Note: borrow checker somehow doesn't accept `self.source` and `self.destination`
                // without using an Option.
                Box::new(Fin(self.source.take().unwrap(),
                             self.destination.take().unwrap()))
            }

            fn buffer(&self, num: usize) -> &BufferAccess {
                assert_eq!(num, 0);
                self.destination.as_ref().unwrap()
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "destination".into()
            }

            fn image(&self, num: usize) -> &ImageAccess {
                assert_eq!(num, 0);
                self.source.as_ref().unwrap()
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "source".into()
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 source: Some(source),
                                                                 destination: Some(destination),
                                                                 source_layout: source_layout,
                                                                 regions: Some(regions),
                                                             }));
        self.prev_cmd_resource(KeyTy::Image,
                               0,
                               false,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_read: true,
                                   ..AccessFlagBits::none()
                               },
                               source_layout,
                               source_layout)?;
        self.prev_cmd_resource(KeyTy::Buffer,
                               0,
                               true,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_write: true,
                                   ..AccessFlagBits::none()
                               },
                               ImageLayout::Undefined,
                               ImageLayout::Undefined)?;
        Ok(())
    }

    /// Calls `vkCmdDispatch` on the builder.
    #[inline]
    pub unsafe fn dispatch(&mut self, dimensions: [u32; 3]) {
        struct Cmd {
            dimensions: [u32; 3],
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdDispatch"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.dispatch(self.dimensions);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { dimensions }));
    }

    /// Calls `vkCmdDispatchIndirect` on the builder.
    #[inline]
    pub unsafe fn dispatch_indirect<B>(&mut self, buffer: B)
                                       -> Result<(), SyncCommandBufferBuilderError>
        where B: BufferAccess + Send + Sync + 'static
    {
        struct Cmd<B> {
            buffer: B,
        }

        impl<P, B> Command<P> for Cmd<B>
            where B: BufferAccess + Send + Sync + 'static
        {
            fn name(&self) -> &'static str {
                "vkCmdDispatchIndirect"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.dispatch_indirect(&self.buffer);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                    where B: BufferAccess + Send + Sync + 'static
                {
                    fn buffer(&self, num: usize) -> &BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "indirect buffer".into()
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { buffer }));
        self.prev_cmd_resource(KeyTy::Buffer,
                               0,
                               false,
                               PipelineStages {
                                   draw_indirect: true,
                                   ..PipelineStages::none()
                               }, // TODO: is draw_indirect correct?
                               AccessFlagBits {
                                   indirect_command_read: true,
                                   ..AccessFlagBits::none()
                               },
                               ImageLayout::Undefined,
                               ImageLayout::Undefined)?;
        Ok(())
    }

    /// Calls `vkCmdDraw` on the builder.
    #[inline]
    pub unsafe fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32,
                       first_instance: u32) {
        struct Cmd {
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdDraw"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.draw(self.vertex_count,
                         self.instance_count,
                         self.first_vertex,
                         self.first_instance);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 vertex_count,
                                                                 instance_count,
                                                                 first_vertex,
                                                                 first_instance,
                                                             }));

    }

    /// Calls `vkCmdDrawIndexed` on the builder.
    #[inline]
    pub unsafe fn draw_indexed(&mut self, index_count: u32, instance_count: u32,
                               first_index: u32, vertex_offset: i32, first_instance: u32) {
        struct Cmd {
            index_count: u32,
            instance_count: u32,
            first_index: u32,
            vertex_offset: i32,
            first_instance: u32,
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdDrawIndexed"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.draw_indexed(self.index_count,
                                 self.instance_count,
                                 self.first_index,
                                 self.vertex_offset,
                                 self.first_instance);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 index_count,
                                                                 instance_count,
                                                                 first_index,
                                                                 vertex_offset,
                                                                 first_instance,
                                                             }));
    }

    /// Calls `vkCmdDrawIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indirect<B>(&mut self, buffer: B, draw_count: u32, stride: u32)
                                   -> Result<(), SyncCommandBufferBuilderError>
        where B: BufferAccess + Send + Sync + 'static
    {
        struct Cmd<B> {
            buffer: B,
            draw_count: u32,
            stride: u32,
        }

        impl<P, B> Command<P> for Cmd<B>
            where B: BufferAccess + Send + Sync + 'static
        {
            fn name(&self) -> &'static str {
                "vkCmdDrawIndirect"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.draw_indirect(&self.buffer, self.draw_count, self.stride);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                    where B: BufferAccess + Send + Sync + 'static
                {
                    fn buffer(&self, num: usize) -> &BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "indirect buffer".into()
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 buffer,
                                                                 draw_count,
                                                                 stride,
                                                             }));
        self.prev_cmd_resource(KeyTy::Buffer,
                               0,
                               false,
                               PipelineStages {
                                   draw_indirect: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   indirect_command_read: true,
                                   ..AccessFlagBits::none()
                               },
                               ImageLayout::Undefined,
                               ImageLayout::Undefined)?;
        Ok(())
    }

    /// Calls `vkCmdDrawIndexedIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indexed_indirect<B>(&mut self, buffer: B, draw_count: u32, stride: u32)
                                           -> Result<(), SyncCommandBufferBuilderError>
        where B: BufferAccess + Send + Sync + 'static
    {
        struct Cmd<B> {
            buffer: B,
            draw_count: u32,
            stride: u32,
        }

        impl<P, B> Command<P> for Cmd<B>
            where B: BufferAccess + Send + Sync + 'static
        {
            fn name(&self) -> &'static str {
                "vkCmdDrawIndexedIndirect"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.draw_indexed_indirect(&self.buffer, self.draw_count, self.stride);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                    where B: BufferAccess + Send + Sync + 'static
                {
                    fn buffer(&self, num: usize) -> &BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "indirect buffer".into()
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 buffer,
                                                                 draw_count,
                                                                 stride,
                                                             }));
        self.prev_cmd_resource(KeyTy::Buffer,
                               0,
                               false,
                               PipelineStages {
                                   draw_indirect: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   indirect_command_read: true,
                                   ..AccessFlagBits::none()
                               },
                               ImageLayout::Undefined,
                               ImageLayout::Undefined)?;
        Ok(())
    }

    /// Calls `vkCmdEndRenderPass` on the builder.
    #[inline]
    pub unsafe fn end_render_pass(&mut self) {
        struct Cmd;

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdEndRenderPass"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.end_render_pass();
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        let mut cmd_lock = self.commands.lock().unwrap();
        cmd_lock.commands.push(Box::new(Cmd));
        debug_assert!(cmd_lock.latest_render_pass_enter.is_some());
        cmd_lock.latest_render_pass_enter = None;
    }

    /// Starts the process of executing secondary command buffers. Returns an intermediate struct
    /// which can be used to add the command buffers.
    #[inline]
    pub unsafe fn execute_commands(&mut self) -> SyncCommandBufferBuilderExecuteCommands<P> {
        SyncCommandBufferBuilderExecuteCommands {
            builder: self,
            inner: UnsafeCommandBufferBuilderExecuteCommands::new(),
            command_buffers: Vec::new(),
        }
    }

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer<B>(&mut self, buffer: B, data: u32)
        where B: BufferAccess + Send + Sync + 'static
    {
        struct Cmd<B> {
            buffer: B,
            data: u32,
        }

        impl<P, B> Command<P> for Cmd<B>
            where B: BufferAccess + Send + Sync + 'static
        {
            fn name(&self) -> &'static str {
                "vkCmdFillBuffer"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.fill_buffer(&self.buffer, self.data);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                    where B: BufferAccess + Send + Sync + 'static
                {
                    fn buffer(&self, num: usize) -> &BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                "destination".into()
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { buffer, data }));
        self.prev_cmd_resource(KeyTy::Buffer,
                               0,
                               true,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_write: true,
                                   ..AccessFlagBits::none()
                               },
                               ImageLayout::Undefined,
                               ImageLayout::Undefined)
            .unwrap();
    }

    /// Calls `vkCmdNextSubpass` on the builder.
    #[inline]
    pub unsafe fn next_subpass(&mut self, subpass_contents: SubpassContents) {
        struct Cmd {
            subpass_contents: SubpassContents,
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdNextSubpass"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.next_subpass(self.subpass_contents);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { subpass_contents }));
    }

    /// Calls `vkCmdPushConstants` on the builder.
    #[inline]
    pub unsafe fn push_constants<Pl, D>(&mut self, pipeline_layout: Pl, stages: ShaderStages,
                                        offset: u32, size: u32, data: &D)
        where Pl: PipelineLayoutAbstract + Send + Sync + 'static,
              D: ?Sized + Send + Sync + 'static
    {
        struct Cmd<Pl> {
            pipeline_layout: Pl,
            stages: ShaderStages,
            offset: u32,
            size: u32,
            data: Box<[u8]>,
        }

        impl<P, Pl> Command<P> for Cmd<Pl>
            where Pl: PipelineLayoutAbstract + Send + Sync + 'static
        {
            fn name(&self) -> &'static str {
                "vkCmdPushConstants"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.push_constants::<_, [u8]>(&self.pipeline_layout,
                                              self.stages,
                                              self.offset,
                                              self.size,
                                              &self.data);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<Pl>(Pl);
                impl<Pl> FinalCommand for Fin<Pl>
                    where Pl: Send + Sync + 'static
                {
                }
                Box::new(Fin(self.pipeline_layout))
            }
        }

        debug_assert!(mem::size_of_val(data) >= size as usize);

        let mut out = Vec::with_capacity(size as usize);
        ptr::copy::<u8>(data as *const D as *const u8,
                        out.as_mut_ptr(),
                        size as usize);
        out.set_len(size as usize);

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 pipeline_layout,
                                                                 stages,
                                                                 offset,
                                                                 size,
                                                                 data: out.into(),
                                                             }));
    }

    /// Calls `vkCmdResetEvent` on the builder.
    #[inline]
    pub unsafe fn reset_event(&mut self, event: Arc<Event>, stages: PipelineStages) {
        struct Cmd {
            event: Arc<Event>,
            stages: PipelineStages,
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdResetEvent"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.reset_event(&self.event, self.stages);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin(Arc<Event>);
                impl FinalCommand for Fin {
                }
                Box::new(Fin(self.event))
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { event, stages }));
    }

    /// Calls `vkCmdSetBlendConstants` on the builder.
    #[inline]
    pub unsafe fn set_blend_constants(&mut self, constants: [f32; 4]) {
        struct Cmd {
            constants: [f32; 4],
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetBlendConstants"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.set_blend_constants(self.constants);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { constants }));
    }

    /// Calls `vkCmdSetDepthBias` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias(&mut self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        struct Cmd {
            constant_factor: f32,
            clamp: f32,
            slope_factor: f32,
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetDepthBias"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.set_depth_bias(self.constant_factor, self.clamp, self.slope_factor);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 constant_factor,
                                                                 clamp,
                                                                 slope_factor,
                                                             }));
    }

    /// Calls `vkCmdSetDepthBounds` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds(&mut self, min: f32, max: f32) {
        struct Cmd {
            min: f32,
            max: f32,
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetDepthBounds"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.set_depth_bounds(self.min, self.max);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { min, max }));
    }

    /// Calls `vkCmdSetEvent` on the builder.
    #[inline]
    pub unsafe fn set_event(&mut self, event: Arc<Event>, stages: PipelineStages) {
        struct Cmd {
            event: Arc<Event>,
            stages: PipelineStages,
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetEvent"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.set_event(&self.event, self.stages);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin(Arc<Event>);
                impl FinalCommand for Fin {
                }
                Box::new(Fin(self.event))
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { event, stages }));
    }

    /// Calls `vkCmdSetLineWidth` on the builder.
    #[inline]
    pub unsafe fn set_line_width(&mut self, line_width: f32) {
        struct Cmd {
            line_width: f32,
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetLineWidth"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.set_line_width(self.line_width);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { line_width }));
    }

    // TODO: stencil states

    /// Calls `vkCmdSetScissor` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_scissor<I>(&mut self, first_scissor: u32, scissors: I)
        where I: Iterator<Item = Scissor> + Send + Sync + 'static
    {
        struct Cmd<I> {
            first_scissor: u32,
            scissors: Option<I>,
        }

        impl<P, I> Command<P> for Cmd<I>
            where I: Iterator<Item = Scissor>
        {
            fn name(&self) -> &'static str {
                "vkCmdSetScissor"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.set_scissor(self.first_scissor, self.scissors.take().unwrap());
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 first_scissor,
                                                                 scissors: Some(scissors),
                                                             }));
    }

    /// Calls `vkCmdSetViewport` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_viewport<I>(&mut self, first_viewport: u32, viewports: I)
        where I: Iterator<Item = Viewport> + Send + Sync + 'static
    {
        struct Cmd<I> {
            first_viewport: u32,
            viewports: Option<I>,
        }

        impl<P, I> Command<P> for Cmd<I>
            where I: Iterator<Item = Viewport>
        {
            fn name(&self) -> &'static str {
                "vkCmdSetViewport"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.set_viewport(self.first_viewport, self.viewports.take().unwrap());
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                Box::new(())
            }
        }

        self.commands.lock().unwrap().commands.push(Box::new(Cmd {
                                                                 first_viewport,
                                                                 viewports: Some(viewports),
                                                             }));
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    #[inline]
    pub unsafe fn update_buffer<B, D>(&mut self, buffer: B, data: D)
        where B: BufferAccess + Send + Sync + 'static,
              D: Send + Sync + 'static
    {
        struct Cmd<B, D> {
            buffer: B,
            data: D,
        }

        impl<P, B, D> Command<P> for Cmd<B, D>
            where B: BufferAccess + Send + Sync + 'static,
                  D: Send + Sync + 'static
        {
            fn name(&self) -> &'static str {
                "vkCmdUpdateBuffer"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.update_buffer(&self.buffer, &self.data);
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                    where B: BufferAccess + Send + Sync + 'static
                {
                    fn buffer(&self, num: usize) -> &BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                "destination".into()
            }
        }

        self.commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd { buffer, data }));
        self.prev_cmd_resource(KeyTy::Buffer,
                               0,
                               true,
                               PipelineStages {
                                   transfer: true,
                                   ..PipelineStages::none()
                               },
                               AccessFlagBits {
                                   transfer_write: true,
                                   ..AccessFlagBits::none()
                               },
                               ImageLayout::Undefined,
                               ImageLayout::Undefined)
            .unwrap();
    }
}

unsafe impl<P> DeviceOwned for SyncCommandBufferBuilder<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

pub struct SyncCommandBufferBuilderBindDescriptorSets<'b, P: 'b> {
    builder: &'b mut SyncCommandBufferBuilder<P>,
    inner: SmallVec<[Box<DescriptorSet + Send + Sync>; 12]>,
}

impl<'b, P> SyncCommandBufferBuilderBindDescriptorSets<'b, P> {
    /// Adds a descriptor set to the list.
    #[inline]
    pub fn add<S>(&mut self, set: S)
        where S: DescriptorSet + Send + Sync + 'static
    {
        self.inner.push(Box::new(set));
    }

    #[inline]
    pub unsafe fn submit<Pl, I>(self, graphics: bool, pipeline_layout: Pl, first_binding: u32,
                                dynamic_offsets: I)
                                -> Result<(), SyncCommandBufferBuilderError>
        where Pl: PipelineLayoutAbstract + Send + Sync + 'static,
              I: Iterator<Item = u32> + Send + Sync + 'static
    {
        struct Cmd<Pl, I> {
            inner: SmallVec<[Box<DescriptorSet + Send + Sync>; 12]>,
            graphics: bool,
            pipeline_layout: Pl,
            first_binding: u32,
            dynamic_offsets: Option<I>,
        }

        impl<P, Pl, I> Command<P> for Cmd<Pl, I>
            where Pl: PipelineLayoutAbstract,
                  I: Iterator<Item = u32>
        {
            fn name(&self) -> &'static str {
                "vkCmdBindDescriptorSets"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.bind_descriptor_sets(self.graphics,
                                         &self.pipeline_layout,
                                         self.first_binding,
                                         self.inner.iter().map(|s| s.inner()),
                                         self.dynamic_offsets.take().unwrap());
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin(SmallVec<[Box<DescriptorSet + Send + Sync>; 12]>);
                impl FinalCommand for Fin {
                    fn buffer(&self, mut num: usize) -> &BufferAccess {
                        for set in self.0.iter() {
                            if let Some(buf) = set.buffer(num) {
                                return buf.0;
                            }
                            num -= set.num_buffers();
                        }
                        panic!()
                    }
                    fn image(&self, mut num: usize) -> &ImageAccess {
                        for set in self.0.iter() {
                            if let Some(img) = set.image(num) {
                                return img.0.parent();
                            }
                            num -= set.num_images();
                        }
                        panic!()
                    }
                }
                Box::new(Fin(self.inner))
            }

            fn buffer(&self, mut num: usize) -> &BufferAccess {
                for set in self.inner.iter() {
                    if let Some(buf) = set.buffer(num) {
                        return buf.0;
                    }
                    num -= set.num_buffers();
                }
                panic!()
            }

            fn buffer_name(&self, mut num: usize) -> Cow<'static, str> {
                for (set_num, set) in self.inner.iter().enumerate() {
                    if let Some(buf) = set.buffer(num) {
                        return format!("Buffer bound to descriptor {} of set {}", buf.1, set_num).into();
                    }
                    num -= set.num_buffers();
                }
                panic!()
            }

            fn image(&self, mut num: usize) -> &ImageAccess {
                for set in self.inner.iter() {
                    if let Some(img) = set.image(num) {
                        return img.0.parent();
                    }
                    num -= set.num_images();
                }
                panic!()
            }

            fn image_name(&self, mut num: usize) -> Cow<'static, str> {
                for (set_num, set) in self.inner.iter().enumerate() {
                    if let Some(img) = set.image(num) {
                        return format!("Image bound to descriptor {} of set {}", img.1, set_num).into();
                    }
                    num -= set.num_images();
                }
                panic!()
            }
        }

        let all_buffers = {
            let mut all_buffers = Vec::new();
            for ds in self.inner.iter() {
                for buf_num in 0 .. ds.num_buffers() {
                    let desc = ds.descriptor(ds.buffer(buf_num).unwrap().1 as usize).unwrap();
                    let write = !desc.readonly;
                    let (stages, access) = desc.pipeline_stages_and_access();
                    all_buffers.push((write, stages, access));
                }
            }
            all_buffers
        };

        let all_images = {
            let mut all_images = Vec::new();
            for ds in self.inner.iter() {
                for img_num in 0 .. ds.num_images() {
                    let (image_view, desc_num) = ds.image(img_num).unwrap();
                    let desc = ds.descriptor(desc_num as usize).unwrap();
                    let write = !desc.readonly;
                    let (stages, access) = desc.pipeline_stages_and_access();
                    let mut ignore_me_hack = false;
                    let layout = match desc.ty {
                        DescriptorDescTy::CombinedImageSampler(_) => {
                            image_view.descriptor_set_combined_image_sampler_layout()
                        },
                        DescriptorDescTy::Image(ref img) => {
                            if img.sampled {
                                image_view.descriptor_set_sampled_image_layout()
                            } else {
                                image_view.descriptor_set_storage_image_layout()
                            }
                        },
                        DescriptorDescTy::InputAttachment { .. } => {
                            // FIXME: This is tricky. Since we read from the input attachment
                            // and this input attachment is being written in an earlier pass,
                            // vulkano will think that it needs to put a pipeline barrier and will
                            // return a `Conflict` error. For now as a work-around we simply ignore
                            // input attachments.
                            ignore_me_hack = true;
                            image_view.descriptor_set_input_attachment_layout()
                        },
                        _ => panic!("Tried to bind an image to a non-image descriptor")
                    };
                    all_images.push((write, stages, access, layout, ignore_me_hack));
                }
            }
            all_images
        };

        self.builder
            .commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd {
                               inner: self.inner,
                               graphics,
                               pipeline_layout,
                               first_binding,
                               dynamic_offsets: Some(dynamic_offsets),
                           }));

        for (n, (write, stages, access)) in all_buffers.into_iter().enumerate() {
            self.builder
                .prev_cmd_resource(KeyTy::Buffer,
                                   n, write, stages, access,
                                   ImageLayout::Undefined,
                                   ImageLayout::Undefined)?;
        }

        for (n, (write, stages, access, layout, ignore_me_hack)) in all_images.into_iter().enumerate() {
            if ignore_me_hack { continue; }
            self.builder
                .prev_cmd_resource(KeyTy::Image,
                                   n, write, stages, access,
                                   layout,
                                   layout)?;
        }

        Ok(())
    }
}

/// Prototype for a `vkCmdBindVertexBuffers`.
pub struct SyncCommandBufferBuilderBindVertexBuffer<'a, P: 'a> {
    builder: &'a mut SyncCommandBufferBuilder<P>,
    inner: UnsafeCommandBufferBuilderBindVertexBuffer,
    buffers: Vec<Box<BufferAccess + Send + Sync>>,
}

impl<'a, P> SyncCommandBufferBuilderBindVertexBuffer<'a, P> {
    /// Adds a buffer to the list.
    #[inline]
    pub fn add<B>(&mut self, buffer: B)
        where B: BufferAccess + Send + Sync + 'static
    {
        self.inner.add(&buffer);
        self.buffers.push(Box::new(buffer));
    }

    #[inline]
    pub unsafe fn submit(self, first_binding: u32) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            first_binding: u32,
            inner: Option<UnsafeCommandBufferBuilderBindVertexBuffer>,
            buffers: Vec<Box<BufferAccess + Send + Sync>>,
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdBindVertexBuffers"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.bind_vertex_buffers(self.first_binding, self.inner.take().unwrap());
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin(Vec<Box<BufferAccess + Send + Sync>>);
                impl FinalCommand for Fin {
                    fn buffer(&self, num: usize) -> &BufferAccess {
                        &self.0[num]
                    }
                }
                Box::new(Fin(self.buffers))
            }

            fn buffer(&self, num: usize) -> &BufferAccess {
                &self.buffers[num]
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                format!("Buffer #{}", num).into()
            }
        }

        let num_buffers = self.buffers.len();

        self.builder
            .commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd {
                               first_binding,
                               inner: Some(self.inner),
                               buffers: self.buffers,
                           }));

        for n in 0 .. num_buffers {
            self.builder
                .prev_cmd_resource(KeyTy::Buffer,
                                   n,
                                   false,
                                   PipelineStages {
                                       vertex_input: true,
                                       ..PipelineStages::none()
                                   },
                                   AccessFlagBits {
                                       vertex_attribute_read: true,
                                       ..AccessFlagBits::none()
                                   },
                                   ImageLayout::Undefined,
                                   ImageLayout::Undefined)?;
        }

        Ok(())
    }
}

/// Prototype for a `vkCmdExecuteCommands`.
// FIXME: synchronization not implemented yet
pub struct SyncCommandBufferBuilderExecuteCommands<'a, P: 'a> {
    builder: &'a mut SyncCommandBufferBuilder<P>,
    inner: UnsafeCommandBufferBuilderExecuteCommands,
    command_buffers: Vec<Box<Any + Send + Sync>>,
}

impl<'a, P> SyncCommandBufferBuilderExecuteCommands<'a, P> {
    /// Adds a command buffer to the list.
    #[inline]
    pub fn add<C>(&mut self, command_buffer: C)
        where C: CommandBuffer + Send + Sync + 'static
    {
        self.inner.add(&command_buffer);
        self.command_buffers.push(Box::new(command_buffer) as Box<_>);
    }

    #[inline]
    pub unsafe fn submit(self) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            inner: Option<UnsafeCommandBufferBuilderExecuteCommands>,
            command_buffers: Vec<Box<Any + Send + Sync>>,
        }

        impl<P> Command<P> for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdExecuteCommands"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>) {
                out.execute_commands(self.inner.take().unwrap());
            }

            fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync> {
                struct Fin(Vec<Box<Any + Send + Sync>>);
                impl FinalCommand for Fin {
                }
                Box::new(Fin(self.command_buffers))
            }
        }

        self.builder
            .commands
            .lock()
            .unwrap()
            .commands
            .push(Box::new(Cmd {
                               inner: Some(self.inner),
                               command_buffers: self.command_buffers,
                           }));
        Ok(())
    }
}

pub struct SyncCommandBuffer<P> {
    // The actual Vulkan command buffer.
    inner: UnsafeCommandBuffer<P>,

    // State of all the resources used by this command buffer.
    resources: FnvHashMap<CbKey<'static>, ResourceFinalState>,

    // List of commands used by the command buffer. Used to hold the various resources that are
    // being used. Each element of `resources` has a copy of this `Arc`, but we need to keep one
    // here in case `resources` is empty.
    commands: Arc<Mutex<Vec<Box<FinalCommand + Send + Sync>>>>,
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
trait FinalCommand {
    // Gives access to the `num`th buffer used by the command.
    fn buffer(&self, num: usize) -> &BufferAccess {
        panic!()
    }
    // Gives access to the `num`th image used by the command.
    fn image(&self, num: usize) -> &ImageAccess {
        panic!()
    }
}

impl FinalCommand for () {
}

// Equivalent of `BuilderKey` for a finished command buffer.
//
// In addition to this, it also add other variants. TODO: document
enum CbKey<'a> {
    // The resource is held in the list of commands.
    Command {
        // Same `Arc` as the `SyncCommandBufferBuilder`.
        commands: Arc<Mutex<Vec<Box<FinalCommand + Send + Sync>>>>,
        // Index of the command that holds the resource within `commands`.
        command_id: usize,
        // Type of the resource.
        resource_ty: KeyTy,
        // Index of the resource within the command.
        resource_index: usize,
    },

    // Temporary key that holds a reference to a buffer. Should never be stored in the list of
    // resources of `SyncCommandBuffer`.
    BufferRef(&'a BufferAccess),
    // Temporary key that holds a reference to an image. Should never be stored in the list of
    // resources of `SyncCommandBuffer`.
    ImageRef(&'a ImageAccess),
}

// The `CbKey::Command` variants implements `Send` and `Sync`, but not the other two variants
// because it would be too constraining.
// Since only `CbKey::Command` must be stored in the resources hashmap, we force-implement `Send`
// and `Sync` so that the hashmap itself implements `Send` and `Sync`.
unsafe impl<'a> Send for CbKey<'a> {
}
unsafe impl<'a> Sync for CbKey<'a> {
}

impl<'a> CbKey<'a> {
    #[inline]
    fn conflicts_buffer_all(&self, commands_lock: Option<&Vec<Box<FinalCommand + Send + Sync>>>,
                            buf: &BufferAccess)
                            -> bool {
        match *self {
            CbKey::Command {
                ref commands,
                command_id,
                resource_ty,
                resource_index,
            } => {
                let lock = if commands_lock.is_none() {
                    Some(commands.lock().unwrap())
                } else {
                    None
                };
                let commands_lock = commands_lock.unwrap_or_else(|| lock.as_ref().unwrap());

                // TODO: put the conflicts_* methods directly on the FinalCommand trait to avoid an indirect call?
                match resource_ty {
                    KeyTy::Buffer => {
                        let c = &commands_lock[command_id];
                        c.buffer(resource_index).conflicts_buffer_all(buf)
                    },
                    KeyTy::Image => {
                        let c = &commands_lock[command_id];
                        c.image(resource_index).conflicts_buffer_all(buf)
                    },
                }
            },

            CbKey::BufferRef(b) => b.conflicts_buffer_all(buf),
            CbKey::ImageRef(i) => i.conflicts_buffer_all(buf),
        }
    }

    #[inline]
    fn conflicts_image_all(&self, commands_lock: Option<&Vec<Box<FinalCommand + Send + Sync>>>,
                           img: &ImageAccess)
                           -> bool {
        match *self {
            CbKey::Command {
                ref commands,
                command_id,
                resource_ty,
                resource_index,
            } => {
                let lock = if commands_lock.is_none() {
                    Some(commands.lock().unwrap())
                } else {
                    None
                };
                let commands_lock = commands_lock.unwrap_or_else(|| lock.as_ref().unwrap());

                // TODO: put the conflicts_* methods directly on the Command trait to avoid an indirect call?
                match resource_ty {
                    KeyTy::Buffer => {
                        let c = &commands_lock[command_id];
                        c.buffer(resource_index).conflicts_image_all(img)
                    },
                    KeyTy::Image => {
                        let c = &commands_lock[command_id];
                        c.image(resource_index).conflicts_image_all(img)
                    },
                }
            },

            CbKey::BufferRef(b) => b.conflicts_image_all(img),
            CbKey::ImageRef(i) => i.conflicts_image_all(img),
        }
    }
}

impl<'a> PartialEq for CbKey<'a> {
    #[inline]
    fn eq(&self, other: &CbKey) -> bool {
        match *self {
            CbKey::BufferRef(a) => {
                other.conflicts_buffer_all(None, a)
            },
            CbKey::ImageRef(a) => {
                other.conflicts_image_all(None, a)
            },
            CbKey::Command {
                ref commands,
                command_id,
                resource_ty,
                resource_index,
            } => {
                let commands_lock = commands.lock().unwrap();

                match resource_ty {
                    KeyTy::Buffer => {
                        let c = &commands_lock[command_id];
                        other.conflicts_buffer_all(Some(&commands_lock), c.buffer(resource_index))
                    },
                    KeyTy::Image => {
                        let c = &commands_lock[command_id];
                        other.conflicts_image_all(Some(&commands_lock), c.image(resource_index))
                    },
                }
            },
        }
    }
}

impl<'a> Eq for CbKey<'a> {
}

impl<'a> Hash for CbKey<'a> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            CbKey::Command {
                ref commands,
                command_id,
                resource_ty,
                resource_index,
            } => {
                let commands_lock = commands.lock().unwrap();

                match resource_ty {
                    KeyTy::Buffer => {
                        let c = &commands_lock[command_id];
                        c.buffer(resource_index).conflict_key_all().hash(state)
                    },
                    KeyTy::Image => {
                        let c = &commands_lock[command_id];
                        c.image(resource_index).conflict_key_all().hash(state)
                    },
                }
            },

            CbKey::BufferRef(buf) => buf.conflict_key_all().hash(state),
            CbKey::ImageRef(img) => img.conflict_key_all().hash(state),
        }
    }
}

// TODO: should we really implement this trait on this type?
unsafe impl<P> CommandBuffer for SyncCommandBuffer<P> {
    type PoolAlloc = P;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::PoolAlloc> {
        &self.inner
    }

    fn lock_submit(&self, future: &GpuFuture, queue: &Queue)
                   -> Result<(), CommandBufferExecError> {
        // TODO: if at any point we return an error, we can't recover

        let commands_lock = self.commands.lock().unwrap();

        for (key, entry) in self.resources.iter() {
            let (commands, command_id, resource_ty, resource_index) = match *key {
                CbKey::Command {
                    ref commands,
                    command_id,
                    resource_ty,
                    resource_index,
                } => {
                    (commands, command_id, resource_ty, resource_index)
                },
                _ => unreachable!(),
            };

            match resource_ty {
                KeyTy::Buffer => {
                    let cmd = &commands_lock[command_id];
                    let buf = cmd.buffer(resource_index);

                    let prev_err = match future.check_buffer_access(&buf, entry.exclusive, queue) {
                        Ok(_) => {
                            unsafe {
                                buf.increase_gpu_lock();
                            }
                            continue;
                        },
                        Err(err) => err,
                    };

                    match (buf.try_gpu_lock(entry.exclusive, queue), prev_err) {
                        (Ok(_), _) => (),
                        (Err(err), AccessCheckError::Unknown) => return Err(err.into()),
                        (_, AccessCheckError::Denied(err)) => return Err(err.into()),
                    }
                },
                KeyTy::Image => {
                    let cmd = &commands_lock[command_id];
                    let img = cmd.image(resource_index);

                    let prev_err = match future.check_image_access(img, entry.initial_layout,
                                                                   entry.exclusive, queue)
                    {
                        Ok(_) => {
                            unsafe { img.increase_gpu_lock(); }
                            continue;
                        },
                        Err(err) => err
                    };

                    match (img.try_gpu_lock(entry.exclusive, queue), prev_err) {
                        (Ok(_), _) => (),
                        (Err(err), AccessCheckError::Unknown) => return Err(err.into()),
                        (_, AccessCheckError::Denied(err)) => return Err(err.into()),
                    }
                },
            }
        }

        // TODO: pipeline barriers if necessary?

        Ok(())
    }

    unsafe fn unlock(&self) {
        let commands_lock = self.commands.lock().unwrap();

        for (key, entry) in self.resources.iter() {
            let (command_id, resource_ty, resource_index) = match *key {
                CbKey::Command {
                    command_id,
                    resource_ty,
                    resource_index,
                    ..
                } => {
                    (command_id, resource_ty, resource_index)
                },
                _ => unreachable!(),
            };

            match resource_ty {
                KeyTy::Buffer => {
                    let cmd = &commands_lock[command_id];
                    let buf = cmd.buffer(resource_index);
                    buf.unlock();
                },
                KeyTy::Image => {
                    let cmd = &commands_lock[command_id];
                    let img = cmd.image(resource_index);
                    img.unlock();
                },
            }
        }
    }

    #[inline]
    fn check_buffer_access(
        &self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
        -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        // TODO: check the queue family

        if let Some(value) = self.resources.get(&CbKey::BufferRef(buffer)) {
            if !value.exclusive && exclusive {
                return Err(AccessCheckError::Unknown);
            }

            return Ok(Some((value.final_stages, value.final_access)));
        }

        Err(AccessCheckError::Unknown)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool,
                          queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        // TODO: check the queue family

        if let Some(value) = self.resources.get(&CbKey::ImageRef(image)) {
            if layout != ImageLayout::Undefined && value.final_layout != layout {
                return Err(AccessCheckError::Denied(AccessError::UnexpectedImageLayout {
                                                        allowed: value.final_layout,
                                                        requested: layout,
                                                    }));
            }

            if !value.exclusive && exclusive {
                return Err(AccessCheckError::Unknown);
            }

            return Ok(Some((value.final_stages, value.final_access)));
        }

        Err(AccessCheckError::Unknown)
    }
}

unsafe impl<P> DeviceOwned for SyncCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}
