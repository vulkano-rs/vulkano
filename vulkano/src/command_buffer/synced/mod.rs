// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Contains `SyncCommandBufferBuilder` and `SyncCommandBuffer`.
//!
//! # How pipeline stages work in Vulkan
//!
//! Imagine you create a command buffer that contains 10 dispatch commands, and submit that command
//! buffer. According to the Vulkan specs, the implementation is free to execute the 10 commands
//! simultaneously.
//!
//! Now imagine that the command buffer contains 10 draw commands instead. Contrary to the dispatch
//! commands, the draw pipeline contains multiple stages: draw indirect, vertex input, vertex shader,
//! ..., fragment shader, late fragment test, color output. When there are multiple stages, the
//! implementations must start and end the stages in order. In other words it can start the draw
//! indirect stage of all 10 commands, then start the vertex input stage of all 10 commands, and so
//! on. But it can't for example start the fragment shader stage of a command before starting the
//! vertex shader stage of another command. Same thing for ending the stages in the right order.
//!
//! Depending on the type of the command, the pipeline stages are different. Compute shaders use the
//! compute stage, while transfer commands use the transfer stage. The compute and transfer stages
//! aren't ordered.
//!
//! When you submit multiple command buffers to a queue, the implementation doesn't do anything in
//! particular and behaves as if the command buffers were appended to one another. Therefore if you
//! submit a command buffer with 10 dispatch commands, followed with another command buffer with 5
//! dispatch commands, then the implementation can perform the 15 commands simultaneously.
//!
//! ## Introducing barriers
//!
//! In some situations this is not the desired behaviour. If you add a command that writes to a
//! buffer followed with another command that reads that buffer, you don't want them to execute
//! simultaneously. Instead you want the second one to wait until the first one is finished. This
//! is done by adding a pipeline barrier between the two commands.
//!
//! A pipeline barriers has a source stage and a destination stage (plus various other things).
//! A barrier represents a split in the list of commands. When you add it, the stages of the commands
//! before the barrier corresponding to the source stage of the barrier, must finish before the
//! stages of the commands after the barrier corresponding to the destination stage of the barrier
//! can start.
//!
//! For example if you add a barrier that transitions from the compute stage to the compute stage,
//! then the compute stage of all the commands before the barrier must end before the compute stage
//! of all the commands after the barrier can start. This is appropriate for the example about
//! writing then reading the same buffer.
//!
//! ## Batching barriers
//!
//! Since barriers are "expensive" (as the queue must block), vulkano attempts to group as many
//! pipeline barriers as possible into one.
//!
//! Adding a command to a sync command buffer builder does not immediately add it to the underlying
//! command buffer builder. Instead the command is added to a queue, and the builder keeps a
//! prototype of a barrier that must be added before the commands in the queue are flushed.
//!
//! Whenever you add a command, the builder will find out whether a barrier is needed before the
//! command. If so, it will try to merge this barrier with the prototype and add the command to the
//! queue. If not possible, the queue will be entirely flushed and the command added to a fresh new
//! queue with a fresh new barrier prototype.

pub use self::builder::CommandBufferState;
pub use self::builder::SetOrPush;
pub use self::builder::StencilOpStateDynamic;
pub use self::builder::StencilStateDynamic;
pub use self::builder::SyncCommandBufferBuilder;
pub use self::builder::SyncCommandBufferBuilderBindDescriptorSets;
pub use self::builder::SyncCommandBufferBuilderBindVertexBuffer;
pub use self::builder::SyncCommandBufferBuilderError;
pub use self::builder::SyncCommandBufferBuilderExecuteCommands;
use crate::buffer::BufferAccess;
use crate::command_buffer::sys::UnsafeCommandBuffer;
use crate::command_buffer::sys::UnsafeCommandBufferBuilder;
use crate::command_buffer::CommandBufferExecError;
use crate::command_buffer::ImageUninitializedSafe;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::device::Queue;
use crate::image::ImageAccess;
use crate::image::ImageLayout;
use crate::sync::AccessCheckError;
use crate::sync::AccessError;
use crate::sync::AccessFlags;
use crate::sync::GpuFuture;
use crate::sync::PipelineMemoryAccess;
use crate::sync::PipelineStages;
use fnv::FnvHashMap;
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

mod builder;

/// Command buffer built from a `SyncCommandBufferBuilder` that provides utilities to handle
/// synchronization.
pub struct SyncCommandBuffer {
    // The actual Vulkan command buffer.
    inner: UnsafeCommandBuffer,

    // List of commands used by the command buffer. Used to hold the various resources that are
    // being used.
    commands: Vec<Box<dyn Command>>,

    // Locations within commands that pipeline barriers were inserted. For debugging purposes.
    // TODO: present only in cfg(debug_assertions)?
    barriers: Vec<usize>,

    // State of all the resources used by this command buffer.
    resources: FnvHashMap<ResourceKey, ResourceFinalState>,

    // Resources and their accesses. Used for executing secondary command buffers in a primary.
    buffers: Vec<(Arc<dyn BufferAccess>, PipelineMemoryAccess)>,
    images: Vec<(
        Arc<dyn ImageAccess>,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
        ImageUninitializedSafe,
    )>,
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
        for state in self.resources.values() {
            let resource_use = &state.resource_uses[0];

            match &resource_use.resource {
                KeyTy::Buffer(buffer) => {
                    // Because try_gpu_lock needs to be called first,
                    // this should never return Ok without first returning Err
                    let prev_err =
                        match future.check_buffer_access(buffer.as_ref(), state.exclusive, queue) {
                            Ok(_) => {
                                unsafe {
                                    buffer.increase_gpu_lock();
                                }
                                locked_resources += 1;
                                continue;
                            }
                            Err(err) => err,
                        };

                    match (buffer.try_gpu_lock(state.exclusive, queue), prev_err) {
                        (Ok(_), _) => (),
                        (Err(err), AccessCheckError::Unknown)
                        | (_, AccessCheckError::Denied(err)) => {
                            ret_value = Err(CommandBufferExecError::AccessError {
                                error: err,
                                command_name: self.commands[resource_use.command_index]
                                    .name()
                                    .into(),
                                command_param: resource_use.name.clone(),
                                command_offset: resource_use.command_index,
                            });
                            break;
                        }
                    };
                }

                KeyTy::Image(image) => {
                    let prev_err = match future.check_image_access(
                        image.as_ref(),
                        state.initial_layout,
                        state.exclusive,
                        queue,
                    ) {
                        Ok(_) => {
                            unsafe {
                                image.increase_gpu_lock();
                            }
                            locked_resources += 1;
                            continue;
                        }
                        Err(err) => err,
                    };

                    match (
                        image.try_gpu_lock(
                            state.exclusive,
                            state.image_uninitialized_safe.is_safe(),
                            state.initial_layout,
                        ),
                        prev_err,
                    ) {
                        (Ok(_), _) => (),
                        (Err(err), AccessCheckError::Unknown)
                        | (_, AccessCheckError::Denied(err)) => {
                            ret_value = Err(CommandBufferExecError::AccessError {
                                error: err,
                                command_name: self.commands[resource_use.command_index]
                                    .name()
                                    .into(),
                                command_param: resource_use.name.clone(),
                                command_offset: resource_use.command_index,
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
            for state in self.resources.values().take(locked_resources) {
                let resource_use = &state.resource_uses[0];

                match &resource_use.resource {
                    KeyTy::Buffer(buffer) => unsafe {
                        buffer.unlock();
                    },
                    KeyTy::Image(image) => {
                        let trans = if state.final_layout != state.initial_layout {
                            Some(state.final_layout)
                        } else {
                            None
                        };
                        unsafe {
                            image.unlock(trans);
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
        for state in self.resources.values() {
            let resource_use = &state.resource_uses[0];

            match &resource_use.resource {
                KeyTy::Buffer(buffer) => {
                    buffer.unlock();
                }
                KeyTy::Image(image) => {
                    let trans = if state.final_layout != state.initial_layout {
                        Some(state.final_layout)
                    } else {
                        None
                    };
                    image.unlock(trans);
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
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        // TODO: check the queue family
        if let Some(value) = self.resources.get(&buffer.into()) {
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
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        // TODO: check the queue family
        if let Some(value) = self.resources.get(&image.into()) {
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
    pub fn buffer(&self, index: usize) -> Option<(&Arc<dyn BufferAccess>, PipelineMemoryAccess)> {
        self.buffers
            .get(index)
            .map(|(buffer, memory)| (buffer, *memory))
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
        &Arc<dyn ImageAccess>,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
        ImageUninitializedSafe,
    )> {
        self.images.get(index).map(
            |(image, memory, start_layout, end_layout, image_uninitialized_safe)| {
                (
                    image,
                    *memory,
                    *start_layout,
                    *end_layout,
                    *image_uninitialized_safe,
                )
            },
        )
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

// Key that identifies a resource. Implements `PartialEq`, `Eq` and `Hash` so that two resources
// that conflict with each other compare equal.
#[derive(Debug, PartialEq, Eq, Hash)]
enum ResourceKey {
    Buffer((u64, u64)),
    Image(u64, Range<u32>, Range<u32>),
}

impl From<&dyn BufferAccess> for ResourceKey {
    #[inline]
    fn from(buffer: &dyn BufferAccess) -> Self {
        Self::Buffer(buffer.conflict_key())
    }
}

impl From<&dyn ImageAccess> for ResourceKey {
    #[inline]
    fn from(image: &dyn ImageAccess) -> Self {
        Self::Image(
            image.conflict_key(),
            image.current_miplevels_access(),
            image.current_layer_levels_access(),
        )
    }
}

// Usage of a resource in a finished command buffer.
#[derive(Clone)]
struct ResourceFinalState {
    // Lists every use of the resource.
    resource_uses: Vec<ResourceUse>,

    // Stages of the last command that uses the resource.
    final_stages: PipelineStages,
    // Access for the last command that uses the resource.
    final_access: AccessFlags,

    // True if the resource is used in exclusive mode.
    exclusive: bool,

    // Layout that an image must be in at the start of the command buffer. Can be `Undefined` if we
    // don't care.
    initial_layout: ImageLayout,

    // Layout the image will be in at the end of the command buffer.
    final_layout: ImageLayout, // TODO: maybe wrap in an Option to mean that the layout doesn't change? because of buffers?

    image_uninitialized_safe: ImageUninitializedSafe,
}

#[derive(Clone)]
struct ResourceUse {
    command_index: usize,
    resource: KeyTy,
    name: Cow<'static, str>,
}

/// Type of resource whose state is to be tracked.
#[derive(Clone)]
enum KeyTy {
    Buffer(Arc<dyn BufferAccess>),
    Image(Arc<dyn ImageAccess>),
}

// Identifies a resource within the list of commands.
#[derive(Clone, Copy, Debug)]
struct ResourceLocation {
    // Index of the command that holds the resource.
    command_id: usize,
    // Index of the resource within the command.
    resource_index: usize,
}

// Trait for single commands within the list of commands.
trait Command: Send + Sync {
    // Returns a user-friendly name for the command, for error reporting purposes.
    fn name(&self) -> &'static str;

    // Sends the command to the `UnsafeCommandBufferBuilder`. Calling this method twice on the same
    // object will likely lead to a panic.
    unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder);
}

impl std::fmt::Debug for dyn Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
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
    use crate::command_buffer::AutoCommandBufferBuilder;
    use crate::command_buffer::CommandBufferLevel;
    use crate::command_buffer::CommandBufferUsage;
    use crate::descriptor_set::layout::DescriptorDesc;
    use crate::descriptor_set::layout::DescriptorSetLayout;
    use crate::descriptor_set::layout::DescriptorType;
    use crate::descriptor_set::PersistentDescriptorSet;
    use crate::descriptor_set::WriteDescriptorSet;
    use crate::device::Device;
    use crate::pipeline::layout::PipelineLayout;
    use crate::pipeline::PipelineBindPoint;
    use crate::sampler::Sampler;
    use crate::shader::ShaderStages;
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
                    CommandBufferLevel::primary(),
                    CommandBufferUsage::MultipleSubmit,
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
                CommandBufferLevel::primary(),
                CommandBufferUsage::MultipleSubmit,
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
                    let mut builder = AutoCommandBufferBuilder::secondary_compute(
                        device.clone(),
                        queue.family(),
                        CommandBufferUsage::SimultaneousUse,
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
                    CommandBufferLevel::primary(),
                    CommandBufferUsage::SimultaneousUse,
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
                    CommandBufferLevel::primary(),
                    CommandBufferUsage::SimultaneousUse,
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

    #[test]
    fn vertex_buffer_binding() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let pool = Device::standard_command_pool(&device, queue.family());
            let pool_builder_alloc = pool.alloc(false, 1).unwrap().next().unwrap();
            let mut sync = SyncCommandBufferBuilder::new(
                &pool_builder_alloc.inner(),
                CommandBufferLevel::primary(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();
            let buf =
                CpuAccessibleBuffer::from_data(device, BufferUsage::all(), false, 0u32).unwrap();
            let mut buf_builder = sync.bind_vertex_buffers();
            buf_builder.add(buf);
            buf_builder.submit(1);

            assert!(sync.state().vertex_buffer(0).is_none());
            assert!(sync.state().vertex_buffer(1).is_some());
            assert!(sync.state().vertex_buffer(2).is_none());
        }
    }

    #[test]
    fn descriptor_set_binding() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let pool = Device::standard_command_pool(&device, queue.family());
            let pool_builder_alloc = pool.alloc(false, 1).unwrap().next().unwrap();
            let mut sync = SyncCommandBufferBuilder::new(
                &pool_builder_alloc.inner(),
                CommandBufferLevel::primary(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();
            let set_layout = DescriptorSetLayout::new(
                device.clone(),
                [Some(DescriptorDesc {
                    ty: DescriptorType::Sampler,
                    descriptor_count: 1,
                    variable_count: false,
                    stages: ShaderStages::all(),
                    immutable_samplers: Vec::new(),
                })],
            )
            .unwrap();
            let pipeline_layout =
                PipelineLayout::new(device.clone(), [set_layout.clone(), set_layout.clone()], [])
                    .unwrap();

            let set = PersistentDescriptorSet::new(
                set_layout.clone(),
                [WriteDescriptorSet::sampler(
                    0,
                    Sampler::simple_repeat_linear(device.clone()).unwrap(),
                )],
            )
            .unwrap();

            let mut set_builder = sync.bind_descriptor_sets();
            set_builder.add(set.clone());
            set_builder.submit(PipelineBindPoint::Graphics, pipeline_layout.clone(), 1);

            assert!(sync
                .state()
                .descriptor_set(PipelineBindPoint::Compute, 0)
                .is_none());
            assert!(sync
                .state()
                .descriptor_set(PipelineBindPoint::Graphics, 0)
                .is_none());
            assert!(sync
                .state()
                .descriptor_set(PipelineBindPoint::Graphics, 1)
                .is_some());
            assert!(sync
                .state()
                .descriptor_set(PipelineBindPoint::Graphics, 2)
                .is_none());

            let mut set_builder = sync.bind_descriptor_sets();
            set_builder.add(set);
            set_builder.submit(PipelineBindPoint::Graphics, pipeline_layout, 0);

            assert!(sync
                .state()
                .descriptor_set(PipelineBindPoint::Graphics, 0)
                .is_some());
            assert!(sync
                .state()
                .descriptor_set(PipelineBindPoint::Graphics, 1)
                .is_some());

            let pipeline_layout = PipelineLayout::new(
                device.clone(),
                [
                    DescriptorSetLayout::new(device.clone(), []).unwrap(),
                    set_layout.clone(),
                ],
                [],
            )
            .unwrap();

            let set = PersistentDescriptorSet::new(
                set_layout.clone(),
                [WriteDescriptorSet::sampler(
                    0,
                    Sampler::simple_repeat_linear(device.clone()).unwrap(),
                )],
            )
            .unwrap();

            let mut set_builder = sync.bind_descriptor_sets();
            set_builder.add(set);
            set_builder.submit(PipelineBindPoint::Graphics, pipeline_layout, 1);

            assert!(sync
                .state()
                .descriptor_set(PipelineBindPoint::Graphics, 0)
                .is_none());
            assert!(sync
                .state()
                .descriptor_set(PipelineBindPoint::Graphics, 1)
                .is_some());
        }
    }
}
