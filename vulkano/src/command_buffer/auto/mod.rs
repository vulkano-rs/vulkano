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

pub use self::builder::{
    AutoCommandBufferBuilder, CommandBufferBeginError, CommandBufferBuildError,
};
pub(in crate::command_buffer) use self::builder::{
    BeginRenderPassState, BeginRenderingState, QueryState, RenderPassState,
    RenderPassStateAttachments, RenderPassStateType, SetOrPush,
};
use super::{
    allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    sys::{UnsafeCommandBuffer, UnsafeCommandBufferBuilder},
    CommandBufferExecError, CommandBufferInheritanceInfo, CommandBufferResourcesUsage,
    CommandBufferState, CommandBufferUsage, PrimaryCommandBufferAbstract, ResourceInCommand,
    SecondaryCommandBufferAbstract, SecondaryCommandBufferResourcesUsage, SecondaryResourceUseRef,
};
use crate::{
    buffer::Subbuffer,
    device::{Device, DeviceOwned},
    image::{ImageAccess, ImageLayout, ImageSubresourceRange},
    sync::PipelineMemoryAccess,
    DeviceSize, VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use std::{
    fmt::{Debug, Error as FmtError, Formatter},
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

mod builder;

pub struct PrimaryAutoCommandBuffer<A = StandardCommandBufferAllocator>
where
    A: CommandBufferAllocator,
{
    inner: UnsafeCommandBuffer<A>,
    _keep_alive_objects:
        Vec<Box<dyn Fn(&mut UnsafeCommandBufferBuilder<A>) + Send + Sync + 'static>>,
    resources_usage: CommandBufferResourcesUsage,
    state: Mutex<CommandBufferState>,
}

unsafe impl<A> VulkanObject for PrimaryAutoCommandBuffer<A>
where
    A: CommandBufferAllocator,
{
    type Handle = ash::vk::CommandBuffer;

    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl<A> DeviceOwned for PrimaryAutoCommandBuffer<A>
where
    A: CommandBufferAllocator,
{
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<A> PrimaryCommandBufferAbstract for PrimaryAutoCommandBuffer<A>
where
    A: CommandBufferAllocator,
{
    fn usage(&self) -> CommandBufferUsage {
        self.inner.usage()
    }

    fn state(&self) -> MutexGuard<'_, CommandBufferState> {
        self.state.lock()
    }

    fn resources_usage(&self) -> &CommandBufferResourcesUsage {
        &self.resources_usage
    }
}

pub struct SecondaryAutoCommandBuffer<A = StandardCommandBufferAllocator>
where
    A: CommandBufferAllocator,
{
    inner: UnsafeCommandBuffer<A>,
    _keep_alive_objects:
        Vec<Box<dyn Fn(&mut UnsafeCommandBufferBuilder<A>) + Send + Sync + 'static>>,
    resources_usage: SecondaryCommandBufferResourcesUsage,
    submit_state: SubmitState,
}

unsafe impl<A> VulkanObject for SecondaryAutoCommandBuffer<A>
where
    A: CommandBufferAllocator,
{
    type Handle = ash::vk::CommandBuffer;

    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl<A> DeviceOwned for SecondaryAutoCommandBuffer<A>
where
    A: CommandBufferAllocator,
{
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<A> SecondaryCommandBufferAbstract for SecondaryAutoCommandBuffer<A>
where
    A: CommandBufferAllocator,
{
    fn usage(&self) -> CommandBufferUsage {
        self.inner.usage()
    }

    fn inheritance_info(&self) -> &CommandBufferInheritanceInfo {
        self.inner.inheritance_info().as_ref().unwrap()
    }

    fn lock_record(&self) -> Result<(), CommandBufferExecError> {
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                let was_already_submitted = already_submitted.swap(true, Ordering::SeqCst);
                if was_already_submitted {
                    return Err(CommandBufferExecError::OneTimeSubmitAlreadySubmitted);
                }
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let already_in_use = in_use.swap(true, Ordering::SeqCst);
                if already_in_use {
                    return Err(CommandBufferExecError::ExclusiveAlreadyInUse);
                }
            }
            SubmitState::Concurrent => (),
        };

        Ok(())
    }

    unsafe fn unlock(&self) {
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                debug_assert!(already_submitted.load(Ordering::SeqCst));
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let old_val = in_use.swap(false, Ordering::SeqCst);
                debug_assert!(old_val);
            }
            SubmitState::Concurrent => (),
        };
    }

    fn resources_usage(&self) -> &SecondaryCommandBufferResourcesUsage {
        &self.resources_usage
    }
}

// Whether the command buffer can be submitted.
#[derive(Debug)]
enum SubmitState {
    // The command buffer was created with the "SimultaneousUse" flag. Can always be submitted at
    // any time.
    Concurrent,

    // The command buffer can only be submitted once simultaneously.
    ExclusiveUse {
        // True if the command buffer is current in use by the GPU.
        in_use: AtomicBool,
    },

    // The command buffer can only ever be submitted once.
    OneTime {
        // True if the command buffer has already been submitted once and can be no longer be
        // submitted.
        already_submitted: AtomicBool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::command_buffer) struct ResourceUseRef2 {
    pub(in crate::command_buffer) resource_in_command: ResourceInCommand,
    pub(in crate::command_buffer) secondary_use_ref: Option<SecondaryResourceUseRef>,
}

impl From<ResourceInCommand> for ResourceUseRef2 {
    #[inline]
    fn from(resource_in_command: ResourceInCommand) -> Self {
        Self {
            resource_in_command,
            secondary_use_ref: None,
        }
    }
}

/// Type of resource whose state is to be tracked.
#[derive(Clone)]
pub(super) enum Resource {
    Buffer {
        buffer: Subbuffer<[u8]>,
        range: Range<DeviceSize>,
        memory: PipelineMemoryAccess,
    },
    Image {
        image: Arc<dyn ImageAccess>,
        subresource_range: ImageSubresourceRange,
        memory: PipelineMemoryAccess,
        start_layout: ImageLayout,
        end_layout: ImageLayout,
    },
}

struct CommandInfo {
    name: &'static str,
    used_resources: Vec<(ResourceUseRef2, Resource)>,
    render_pass: RenderPassCommand,
}

#[derive(Debug)]
enum RenderPassCommand {
    None,
    Begin,
    End,
}

impl Debug for CommandInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.write_str(self.name)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        buffer::{Buffer, BufferCreateInfo, BufferUsage},
        command_buffer::{
            allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BufferCopy,
            CommandBufferExecError, CommandBufferUsage, CopyBufferInfoTyped, CopyError,
            ExecuteCommandsError, PrimaryCommandBufferAbstract,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator,
            layout::{
                DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
                DescriptorType,
            },
            PersistentDescriptorSet, WriteDescriptorSet,
        },
        device::{Device, DeviceCreateInfo, QueueCreateInfo},
        memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
        pipeline::{layout::PipelineLayoutCreateInfo, PipelineBindPoint, PipelineLayout},
        sampler::{Sampler, SamplerCreateInfo},
        shader::ShaderStages,
        sync::GpuFuture,
    };

    #[test]
    fn basic_creation() {
        let (device, queue) = gfx_dev_and_queue!();

        let allocator = StandardCommandBufferAllocator::new(device, Default::default());

        AutoCommandBufferBuilder::primary(
            &allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
    }

    #[test]
    fn copy_buffer_dimensions() {
        let instance = instance!();

        let physical_device = match instance.enumerate_physical_devices().unwrap().next() {
            Some(p) => p,
            None => return,
        };

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();
        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

        let source = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            [1_u32, 2].iter().copied(),
        )
        .unwrap();

        let destination = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            [0_u32, 10, 20, 3, 4].iter().copied(),
        )
        .unwrap();

        let cb_allocator = StandardCommandBufferAllocator::new(device, Default::default());
        let mut cbb = AutoCommandBufferBuilder::primary(
            &cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cbb.copy_buffer(CopyBufferInfoTyped {
            regions: [BufferCopy {
                src_offset: 0,
                dst_offset: 1,
                size: 2,
                ..Default::default()
            }]
            .into(),
            ..CopyBufferInfoTyped::buffers(source, destination.clone())
        })
        .unwrap();

        let cb = cbb.build().unwrap();

        let future = cb
            .execute(queue)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let result = destination.read().unwrap();

        assert_eq!(*result, [0_u32, 1, 2, 3, 4]);
    }

    #[test]
    fn secondary_nonconcurrent_conflict() {
        let (device, queue) = gfx_dev_and_queue!();

        let cb_allocator = StandardCommandBufferAllocator::new(device, Default::default());

        // Make a secondary CB that doesn't support simultaneous use.
        let builder = AutoCommandBufferBuilder::secondary(
            &cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            Default::default(),
        )
        .unwrap();
        let secondary = builder.build().unwrap();

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                &cb_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();

            // Add the secondary a first time
            builder.execute_commands(secondary.clone()).unwrap();

            // Recording the same non-concurrent secondary command buffer twice into the same
            // primary is an error.
            assert!(matches!(
                builder.execute_commands(secondary.clone()),
                Err(ExecuteCommandsError::ExecError(
                    CommandBufferExecError::ExclusiveAlreadyInUse
                ))
            ));
        }

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                &cb_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();
            builder.execute_commands(secondary.clone()).unwrap();
            let cb1 = builder.build().unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                &cb_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();

            // Recording the same non-concurrent secondary command buffer into multiple
            // primaries is an error.
            assert!(matches!(
                builder.execute_commands(secondary.clone()),
                Err(ExecuteCommandsError::ExecError(
                    CommandBufferExecError::ExclusiveAlreadyInUse
                ))
            ));

            std::mem::drop(cb1);

            // Now that the first cb is dropped, we should be able to record.
            builder.execute_commands(secondary).unwrap();
        }
    }

    #[test]
    fn buffer_self_copy_overlapping() {
        let (device, queue) = gfx_dev_and_queue!();

        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
        let source = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let cb_allocator = StandardCommandBufferAllocator::new(device, Default::default());
        let mut builder = AutoCommandBufferBuilder::primary(
            &cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfoTyped {
                regions: [BufferCopy {
                    src_offset: 0,
                    dst_offset: 2,
                    size: 2,
                    ..Default::default()
                }]
                .into(),
                ..CopyBufferInfoTyped::buffers(source.clone(), source.clone())
            })
            .unwrap();

        let cb = builder.build().unwrap();

        let future = cb
            .execute(queue)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let result = source.read().unwrap();

        assert_eq!(*result, [0_u32, 1, 0, 1]);
    }

    #[test]
    fn buffer_self_copy_not_overlapping() {
        let (device, queue) = gfx_dev_and_queue!();

        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
        let source = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let cb_allocator = StandardCommandBufferAllocator::new(device, Default::default());
        let mut builder = AutoCommandBufferBuilder::primary(
            &cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        assert!(matches!(
            builder.copy_buffer(CopyBufferInfoTyped {
                regions: [BufferCopy {
                    src_offset: 0,
                    dst_offset: 1,
                    size: 2,
                    ..Default::default()
                }]
                .into(),
                ..CopyBufferInfoTyped::buffers(source.clone(), source)
            }),
            Err(CopyError::OverlappingRegions {
                src_region_index: 0,
                dst_region_index: 0,
            })
        ));
    }

    #[test]
    fn secondary_conflicting_writes() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let cb_allocator =
                StandardCommandBufferAllocator::new(device.clone(), Default::default());
            let cbb = AutoCommandBufferBuilder::primary(
                &cb_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            let memory_allocator = StandardMemoryAllocator::new_default(device);
            // Create a tiny test buffer
            let buffer = Buffer::from_data(
                &memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                0u32,
            )
            .unwrap();

            cbb.build()
                .unwrap()
                .execute(queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

            // Two secondary command buffers that both write to the buffer
            let secondary = (0..2)
                .map(|_| {
                    let mut builder = AutoCommandBufferBuilder::secondary(
                        &cb_allocator,
                        queue.queue_family_index(),
                        CommandBufferUsage::SimultaneousUse,
                        Default::default(),
                    )
                    .unwrap();
                    builder
                        .fill_buffer(buffer.clone().into_slice(), 42)
                        .unwrap();
                    builder.build().unwrap()
                })
                .collect::<Vec<_>>();

            {
                let mut builder = AutoCommandBufferBuilder::primary(
                    &cb_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::SimultaneousUse,
                )
                .unwrap();

                // Add both secondary command buffers using separate execute_commands calls.
                secondary.iter().cloned().for_each(|secondary| {
                    builder.execute_commands_unchecked([secondary as _].into_iter().collect());
                });

                let _primary = builder.build().unwrap();
                /*
                let names = primary._commands.iter().map(|c| c.name).collect::<Vec<_>>();

                // Ensure that the builder added a barrier between the two writes
                assert_eq!(&names, &["execute_commands", "execute_commands"]);
                assert_eq!(&primary._barriers, &[0, 1]);
                 */
            }

            {
                let mut builder = AutoCommandBufferBuilder::primary(
                    &cb_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::SimultaneousUse,
                )
                .unwrap();

                // Add a single execute_commands for all secondary command buffers at once
                builder.execute_commands_unchecked(
                    secondary
                        .into_iter()
                        .map(|secondary| secondary as _)
                        .collect(),
                );
            }
        }
    }

    #[test]
    fn vertex_buffer_binding() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let cb_allocator =
                StandardCommandBufferAllocator::new(device.clone(), Default::default());
            let mut sync = AutoCommandBufferBuilder::primary(
                &cb_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            let memory_allocator = StandardMemoryAllocator::new_default(device);
            let buf = Buffer::from_data(
                &memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                0u32,
            )
            .unwrap();
            sync.bind_vertex_buffers_unchecked(1, buf);

            assert!(!sync.builder_state.vertex_buffers.contains_key(&0));
            assert!(sync.builder_state.vertex_buffers.contains_key(&1));
            assert!(!sync.builder_state.vertex_buffers.contains_key(&2));
        }
    }

    #[test]
    fn descriptor_set_binding() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let cb_allocator =
                StandardCommandBufferAllocator::new(device.clone(), Default::default());
            let mut sync = AutoCommandBufferBuilder::primary(
                &cb_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();
            let set_layout = DescriptorSetLayout::new(
                device.clone(),
                DescriptorSetLayoutCreateInfo {
                    bindings: [(
                        0,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::all_graphics(),
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
                        },
                    )]
                    .into(),
                    ..Default::default()
                },
            )
            .unwrap();
            let pipeline_layout = PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: [set_layout.clone(), set_layout.clone()].into(),
                    ..Default::default()
                },
            )
            .unwrap();

            let ds_allocator = StandardDescriptorSetAllocator::new(device.clone());

            let set = PersistentDescriptorSet::new(
                &ds_allocator,
                set_layout.clone(),
                [WriteDescriptorSet::sampler(
                    0,
                    Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear())
                        .unwrap(),
                )],
            )
            .unwrap();

            sync.bind_descriptor_sets_unchecked(
                PipelineBindPoint::Graphics,
                pipeline_layout.clone(),
                1,
                set.clone(),
            );

            assert!(!sync
                .builder_state
                .descriptor_sets
                .get(&PipelineBindPoint::Compute)
                .map_or(false, |state| state.descriptor_sets.contains_key(&0)));
            assert!(!sync
                .builder_state
                .descriptor_sets
                .get(&PipelineBindPoint::Graphics)
                .map_or(false, |state| state.descriptor_sets.contains_key(&0)));
            assert!(sync
                .builder_state
                .descriptor_sets
                .get(&PipelineBindPoint::Graphics)
                .map_or(false, |state| state.descriptor_sets.contains_key(&1)));
            assert!(!sync
                .builder_state
                .descriptor_sets
                .get(&PipelineBindPoint::Graphics)
                .map_or(false, |state| state.descriptor_sets.contains_key(&2)));

            sync.bind_descriptor_sets_unchecked(
                PipelineBindPoint::Graphics,
                pipeline_layout,
                0,
                set,
            );

            assert!(sync
                .builder_state
                .descriptor_sets
                .get(&PipelineBindPoint::Graphics)
                .map_or(false, |state| state.descriptor_sets.contains_key(&0)));
            assert!(sync
                .builder_state
                .descriptor_sets
                .get(&PipelineBindPoint::Graphics)
                .map_or(false, |state| state.descriptor_sets.contains_key(&1)));

            let pipeline_layout = PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: [
                        DescriptorSetLayout::new(device.clone(), Default::default()).unwrap(),
                        set_layout.clone(),
                    ]
                    .into(),
                    ..Default::default()
                },
            )
            .unwrap();

            let set = PersistentDescriptorSet::new(
                &ds_allocator,
                set_layout,
                [WriteDescriptorSet::sampler(
                    0,
                    Sampler::new(device, SamplerCreateInfo::simple_repeat_linear()).unwrap(),
                )],
            )
            .unwrap();

            sync.bind_descriptor_sets_unchecked(
                PipelineBindPoint::Graphics,
                pipeline_layout,
                1,
                set,
            );

            assert!(!sync
                .builder_state
                .descriptor_sets
                .get(&PipelineBindPoint::Graphics)
                .map_or(false, |state| state.descriptor_sets.contains_key(&0)));
            assert!(sync
                .builder_state
                .descriptor_sets
                .get(&PipelineBindPoint::Graphics)
                .map_or(false, |state| state.descriptor_sets.contains_key(&1)));
        }
    }
}
