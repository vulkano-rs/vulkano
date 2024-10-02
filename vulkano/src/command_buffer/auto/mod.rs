//! Contains `AutoCommandBufferBuilder` and the built types `PrimaryCommandBuffer` and
//! `SecondaryCommandBuffer`.
//!
//! # How pipeline stages work in Vulkan
//!
//! Imagine you create a command buffer that contains 10 dispatch commands, and submit that command
//! buffer. According to the Vulkan specs, the implementation is free to execute the 10 commands
//! simultaneously.
//!
//! Now imagine that the command buffer contains 10 draw commands instead. Contrary to the dispatch
//! commands, the draw pipeline contains multiple stages: draw indirect, vertex input, vertex
//! shader, ..., fragment shader, late fragment test, color output. When there are multiple stages,
//! the implementations must start and end the stages in order. In other words it can start the
//! draw indirect stage of all 10 commands, then start the vertex input stage of all 10 commands,
//! and so on. But it can't for example start the fragment shader stage of a command before
//! starting the vertex shader stage of another command. Same thing for ending the stages in the
//! right order.
//!
//! Depending on the type of the command, the pipeline stages are different. Compute shaders use
//! the compute stage, while transfer commands use the transfer stage. The compute and transfer
//! stages aren't ordered.
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
//! A barrier represents a split in the list of commands. When you add it, the stages of the
//! commands before the barrier corresponding to the source stage of the barrier, must finish
//! before the stages of the commands after the barrier corresponding to the destination stage of
//! the barrier can start.
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
//! Adding a command to an `AutoCommandBufferBuilder` does not immediately add it to the underlying
//! command buffer builder. Instead the command is added to a queue, and the builder keeps a
//! prototype of a barrier that must be added before the commands in the queue are flushed.
//!
//! Whenever you add a command, the builder will find out whether a barrier is needed before the
//! command. If so, it will try to merge this barrier with the prototype and add the command to the
//! queue. If not possible, the queue will be entirely flushed and the command added to a fresh new
//! queue with a fresh new barrier prototype.

pub use self::builder::*;
pub(in crate::command_buffer) use self::builder::{
    BeginRenderPassState, BeginRenderingState, QueryState, RenderPassState,
    RenderPassStateAttachments, RenderPassStateType, SetOrPush,
};
use super::{
    sys::{RawCommandBuffer, RawRecordingCommandBuffer},
    CommandBufferInheritanceInfo, CommandBufferLevel, CommandBufferResourcesUsage,
    CommandBufferState, CommandBufferUsage, ResourceInCommand,
    SecondaryCommandBufferResourcesUsage, SecondaryResourceUseRef,
};
use crate::{
    buffer::Subbuffer,
    device::{Device, DeviceOwned},
    image::{Image, ImageLayout, ImageSubresourceRange},
    sync::PipelineStageAccessFlags,
    DeviceSize, ValidationError, VulkanObject,
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

pub struct CommandBuffer {
    inner: RawCommandBuffer,
    // TODO: Remove all of this.
    _keep_alive_objects: Vec<Box<dyn Fn(&mut RawRecordingCommandBuffer) + Send + Sync + 'static>>,
    resources_usage: CommandBufferResourcesUsage,
    secondary_resources_usage: SecondaryCommandBufferResourcesUsage,
    state: Mutex<CommandBufferState>,
    submit_state: SubmitState,
}

unsafe impl VulkanObject for CommandBuffer {
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for CommandBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl Debug for CommandBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("CommandBuffer")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl CommandBuffer {
    /// Returns the inner raw command buffer.
    #[inline]
    pub fn inner(&self) -> &RawCommandBuffer {
        &self.inner
    }

    /// Returns the queue family index of this command buffer.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.inner.queue_family_index()
    }

    /// Returns the level of the command buffer.
    #[inline]
    pub fn level(&self) -> CommandBufferLevel {
        self.inner.level()
    }

    /// Returns the usage of this command buffer.
    #[inline]
    pub fn usage(&self) -> CommandBufferUsage {
        self.inner.usage()
    }

    /// Returns the inheritance info of the command buffer, if it is a secondary command buffer.
    #[inline]
    pub fn inheritance_info(&self) -> Option<&CommandBufferInheritanceInfo> {
        self.inner.inheritance_info()
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, CommandBufferState> {
        debug_assert_eq!(self.level(), CommandBufferLevel::Primary);

        self.state.lock()
    }

    pub(crate) fn resources_usage(&self) -> &CommandBufferResourcesUsage {
        debug_assert_eq!(self.level(), CommandBufferLevel::Primary);

        &self.resources_usage
    }

    /// Checks whether this command buffer is allowed to be recorded to a command buffer,
    /// and if so locks it.
    ///
    /// If you call this function, then you should call `unlock` afterwards.
    pub(crate) fn lock_record(&self) -> Result<(), Box<ValidationError>> {
        debug_assert_eq!(self.level(), CommandBufferLevel::Secondary);

        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                let was_already_submitted = already_submitted.swap(true, Ordering::SeqCst);
                if was_already_submitted {
                    return Err(Box::new(ValidationError {
                        problem: "the command buffer was created with the \
                            `CommandBufferUsage::OneTimeSubmit` usage, but \
                            it was already submitted before"
                            .into(),
                        // vuids?
                        ..Default::default()
                    }));
                }
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let already_in_use = in_use.swap(true, Ordering::SeqCst);
                if already_in_use {
                    return Err(Box::new(ValidationError {
                        problem: "the command buffer was created with the \
                            `CommandBufferUsage::MultipleSubmit` usage, but \
                            it is currently being executed"
                            .into(),
                        // vuids?
                        ..Default::default()
                    }));
                }
            }
            SubmitState::Concurrent => (),
        };

        Ok(())
    }

    /// Unlocks the command buffer. Should be called once for each call to `lock_record`.
    ///
    /// # Safety
    ///
    /// Must not be called if you haven't called `lock_record` before.
    pub(crate) unsafe fn unlock(&self) {
        debug_assert_eq!(self.level(), CommandBufferLevel::Secondary);

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

    pub(crate) fn secondary_resources_usage(&self) -> &SecondaryCommandBufferResourcesUsage {
        &self.secondary_resources_usage
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
#[derive(Clone, Debug)]
pub(super) enum Resource {
    Buffer {
        buffer: Subbuffer<[u8]>,
        range: Range<DeviceSize>,
        memory_access: PipelineStageAccessFlags,
    },
    Image {
        image: Arc<Image>,
        subresource_range: ImageSubresourceRange,
        memory_access: PipelineStageAccessFlags,
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
            allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
            BufferCopy, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage,
            CopyBufferInfoTyped, RecordingCommandBuffer,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator,
            layout::{
                DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
                DescriptorType,
            },
            DescriptorSet, WriteDescriptorSet,
        },
        device::{Device, DeviceCreateInfo, QueueCreateInfo},
        image::sampler::{Sampler, SamplerCreateInfo},
        memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
        pipeline::{layout::PipelineLayoutCreateInfo, PipelineBindPoint, PipelineLayout},
        shader::ShaderStages,
        sync::GpuFuture,
    };
    use std::sync::Arc;

    #[test]
    fn basic_creation() {
        let (device, queue) = gfx_dev_and_queue!();

        let allocator = Arc::new(StandardCommandBufferAllocator::new(
            device,
            Default::default(),
        ));

        RecordingCommandBuffer::new(
            allocator,
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::MultipleSubmit,
                ..Default::default()
            },
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
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let source = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [1_u32, 2].iter().copied(),
        )
        .unwrap();

        let destination = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [0_u32, 10, 20, 3, 4].iter().copied(),
        )
        .unwrap();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device,
            Default::default(),
        ));
        let mut cbb = RecordingCommandBuffer::new(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
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

        let cb = cbb.end().unwrap();

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

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device,
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 1,
                ..Default::default()
            },
        ));

        // Make a secondary CB that doesn't support simultaneous use.
        let builder = RecordingCommandBuffer::new(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferLevel::Secondary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::MultipleSubmit,
                inheritance_info: Some(Default::default()),
                ..Default::default()
            },
        )
        .unwrap();
        let secondary = builder.end().unwrap();

        {
            let mut builder = RecordingCommandBuffer::new(
                cb_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::SimultaneousUse,
                    ..Default::default()
                },
            )
            .unwrap();

            // Add the secondary a first time
            builder.execute_commands(secondary.clone()).unwrap();

            // Recording the same non-concurrent secondary command buffer twice into the same
            // primary is an error.
            assert!(builder.execute_commands(secondary.clone()).is_err());
        }

        {
            let mut builder = RecordingCommandBuffer::new(
                cb_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::SimultaneousUse,
                    ..Default::default()
                },
            )
            .unwrap();
            builder.execute_commands(secondary.clone()).unwrap();
            let cb1 = builder.end().unwrap();

            let mut builder = RecordingCommandBuffer::new(
                cb_allocator,
                queue.queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::SimultaneousUse,
                    ..Default::default()
                },
            )
            .unwrap();

            // Recording the same non-concurrent secondary command buffer into multiple
            // primaries is an error.
            assert!(builder.execute_commands(secondary.clone()).is_err());

            drop(cb1);

            // Now that the first cb is dropped, we should be able to record.
            builder.execute_commands(secondary).unwrap();
        }
    }

    #[test]
    fn buffer_self_copy_overlapping() {
        let (device, queue) = gfx_dev_and_queue!();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let source = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device,
            Default::default(),
        ));
        let mut builder = RecordingCommandBuffer::new(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
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

        let cb = builder.end().unwrap();

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

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let source = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device,
            Default::default(),
        ));
        let mut builder = RecordingCommandBuffer::new(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(builder
            .copy_buffer(CopyBufferInfoTyped {
                regions: [BufferCopy {
                    src_offset: 0,
                    dst_offset: 1,
                    size: 2,
                    ..Default::default()
                }]
                .into(),
                ..CopyBufferInfoTyped::buffers(source.clone(), source)
            })
            .is_err());
    }

    #[test]
    fn secondary_conflicting_writes() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                StandardCommandBufferAllocatorCreateInfo {
                    secondary_buffer_count: 1,
                    ..Default::default()
                },
            ));
            let cbb = RecordingCommandBuffer::new(
                cb_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::OneTimeSubmit,
                    ..Default::default()
                },
            )
            .unwrap();

            let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));
            // Create a tiny test buffer
            let buffer = Buffer::from_data(
                memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                0u32,
            )
            .unwrap();

            cbb.end()
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
                    let mut builder = RecordingCommandBuffer::new(
                        cb_allocator.clone(),
                        queue.queue_family_index(),
                        CommandBufferLevel::Secondary,
                        CommandBufferBeginInfo {
                            usage: CommandBufferUsage::SimultaneousUse,
                            inheritance_info: Some(Default::default()),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    builder
                        .fill_buffer(buffer.clone().into_slice(), 42)
                        .unwrap();
                    builder.end().unwrap()
                })
                .collect::<Vec<_>>();

            {
                let mut builder = RecordingCommandBuffer::new(
                    cb_allocator.clone(),
                    queue.queue_family_index(),
                    CommandBufferLevel::Primary,
                    CommandBufferBeginInfo {
                        usage: CommandBufferUsage::SimultaneousUse,
                        inheritance_info: Some(Default::default()),
                        ..Default::default()
                    },
                )
                .unwrap();

                // Add both secondary command buffers using separate execute_commands calls.
                secondary.iter().cloned().for_each(|secondary| {
                    builder.execute_commands_unchecked([secondary as _].into_iter().collect());
                });

                let _primary = builder.end().unwrap();
                /*
                let names = primary._commands.iter().map(|c| c.name).collect::<Vec<_>>();

                // Ensure that the builder added a barrier between the two writes
                assert_eq!(&names, &["execute_commands", "execute_commands"]);
                assert_eq!(&primary._barriers, &[0, 1]);
                 */
            }

            {
                let mut builder = RecordingCommandBuffer::new(
                    cb_allocator,
                    queue.queue_family_index(),
                    CommandBufferLevel::Primary,
                    CommandBufferBeginInfo {
                        usage: CommandBufferUsage::SimultaneousUse,
                        inheritance_info: Some(Default::default()),
                        ..Default::default()
                    },
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

            let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            ));
            let mut sync = RecordingCommandBuffer::new(
                cb_allocator,
                queue.queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::MultipleSubmit,
                    ..Default::default()
                },
            )
            .unwrap();

            let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));
            let buf = Buffer::from_data(
                memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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

            let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            ));
            let mut sync = RecordingCommandBuffer::new(
                cb_allocator,
                queue.queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::MultipleSubmit,
                    ..Default::default()
                },
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

            let ds_allocator = Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            ));

            let set = DescriptorSet::new(
                ds_allocator.clone(),
                set_layout.clone(),
                [WriteDescriptorSet::sampler(
                    0,
                    Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear())
                        .unwrap(),
                )],
                [],
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

            let set = DescriptorSet::new(
                ds_allocator,
                set_layout,
                [WriteDescriptorSet::sampler(
                    0,
                    Sampler::new(device, SamplerCreateInfo::simple_repeat_linear()).unwrap(),
                )],
                [],
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
