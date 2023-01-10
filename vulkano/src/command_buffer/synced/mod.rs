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
    CommandBufferBuilderState, SetOrPush, StencilOpStateDynamic, StencilStateDynamic,
    SyncCommandBufferBuilder, SyncCommandBufferBuilderBindDescriptorSets,
    SyncCommandBufferBuilderBindVertexBuffer, SyncCommandBufferBuilderError,
    SyncCommandBufferBuilderExecuteCommands,
};
use super::{
    sys::{UnsafeCommandBuffer, UnsafeCommandBufferBuilder},
    CommandBufferResourcesUsage, SecondaryCommandBufferResourcesUsage,
};
use crate::{
    buffer::Subbuffer,
    device::{Device, DeviceOwned},
    image::{ImageAccess, ImageLayout, ImageSubresourceRange},
    sync::PipelineMemoryAccess,
    DeviceSize,
};
use std::{
    fmt::{Debug, Error as FmtError, Formatter},
    ops::Range,
    sync::Arc,
};

mod builder;

/// Command buffer built from a `SyncCommandBufferBuilder` that provides utilities to handle
/// synchronization.
pub struct SyncCommandBuffer {
    // The actual Vulkan command buffer.
    inner: UnsafeCommandBuffer,

    // List of commands used by the command buffer. Used to hold the various resources that are
    // being used.
    _commands: Vec<Box<dyn Command>>,

    // Locations within commands that pipeline barriers were inserted. For debugging purposes.
    // TODO: present only in cfg(debug_assertions)?
    _barriers: Vec<usize>,

    // Resources accessed by this command buffer.
    resources_usage: CommandBufferResourcesUsage,

    // Resources and their accesses. Used for executing secondary command buffers in a primary.
    secondary_resources_usage: SecondaryCommandBufferResourcesUsage,
}

impl SyncCommandBuffer {
    #[inline]
    pub(super) fn resources_usage(&self) -> &CommandBufferResourcesUsage {
        &self.resources_usage
    }

    #[inline]
    pub(super) fn secondary_resources_usage(&self) -> &SecondaryCommandBufferResourcesUsage {
        &self.secondary_resources_usage
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

// Trait for single commands within the list of commands.
pub(super) trait Command: Send + Sync {
    // Returns a user-friendly name for the command, for error reporting purposes.
    fn name(&self) -> &'static str;

    // Sends the command to the `UnsafeCommandBufferBuilder`. Calling this method twice on the same
    // object will likely lead to a panic.
    unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder);
}

impl Debug for dyn Command {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        buffer::{Buffer, BufferAllocateInfo, BufferUsage},
        command_buffer::{
            allocator::{
                CommandBufferAllocator, CommandBufferBuilderAlloc, StandardCommandBufferAllocator,
            },
            sys::CommandBufferBeginInfo,
            AutoCommandBufferBuilder, CommandBufferLevel, CommandBufferUsage, FillBufferInfo,
            PrimaryCommandBufferAbstract,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator,
            layout::{
                DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
                DescriptorType,
            },
            PersistentDescriptorSet, WriteDescriptorSet,
        },
        memory::allocator::StandardMemoryAllocator,
        pipeline::{layout::PipelineLayoutCreateInfo, PipelineBindPoint, PipelineLayout},
        sampler::{Sampler, SamplerCreateInfo},
        shader::ShaderStages,
        sync::GpuFuture,
    };

    #[test]
    fn basic_creation() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let allocator = StandardCommandBufferAllocator::new(device, Default::default());

            let builder_alloc = allocator
                .allocate(queue.queue_family_index(), CommandBufferLevel::Primary, 1)
                .unwrap()
                .next()
                .unwrap();

            SyncCommandBufferBuilder::new(
                builder_alloc.inner(),
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::MultipleSubmit,
                    ..Default::default()
                },
            )
            .unwrap();
        }
    }

    #[test]
    fn secondary_conflicting_writes() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let cb_allocator =
                StandardCommandBufferAllocator::new(device.clone(), Default::default());
            let mut cbb = AutoCommandBufferBuilder::primary(
                &cb_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            let memory_allocator = StandardMemoryAllocator::new_default(device);
            // Create a tiny test buffer
            let buffer = Buffer::from_data(
                &memory_allocator,
                BufferAllocateInfo {
                    buffer_usage: BufferUsage::TRANSFER_DST,
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
                        .fill_buffer(FillBufferInfo {
                            data: 42u32,
                            ..FillBufferInfo::dst_buffer(buffer.clone())
                        })
                        .unwrap();
                    Arc::new(builder.build().unwrap())
                })
                .collect::<Vec<_>>();

            let allocs = cb_allocator
                .allocate(queue.queue_family_index(), CommandBufferLevel::Primary, 2)
                .unwrap()
                .collect::<Vec<_>>();

            {
                let mut builder = SyncCommandBufferBuilder::new(
                    allocs[0].inner(),
                    CommandBufferBeginInfo {
                        usage: CommandBufferUsage::SimultaneousUse,
                        ..Default::default()
                    },
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
                    ._commands
                    .iter()
                    .map(|c| c.name())
                    .collect::<Vec<_>>();

                // Ensure that the builder added a barrier between the two writes
                assert_eq!(&names, &["execute_commands", "execute_commands"]);
                assert_eq!(&primary._barriers, &[0, 1]);
            }

            {
                let mut builder = SyncCommandBufferBuilder::new(
                    allocs[1].inner(),
                    CommandBufferBeginInfo {
                        usage: CommandBufferUsage::SimultaneousUse,
                        ..Default::default()
                    },
                )
                .unwrap();

                // Add a single execute_commands for all secondary command buffers at once
                let mut ec = builder.execute_commands();
                secondary.into_iter().for_each(|secondary| {
                    ec.add(secondary);
                });
                ec.submit().unwrap();
            }
        }
    }

    #[test]
    fn vertex_buffer_binding() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let cb_allocator =
                StandardCommandBufferAllocator::new(device.clone(), Default::default());
            let builder_alloc = cb_allocator
                .allocate(queue.queue_family_index(), CommandBufferLevel::Primary, 1)
                .unwrap()
                .next()
                .unwrap();
            let mut sync = SyncCommandBufferBuilder::new(
                builder_alloc.inner(),
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::MultipleSubmit,
                    ..Default::default()
                },
            )
            .unwrap();

            let memory_allocator = StandardMemoryAllocator::new_default(device);
            let buf = Buffer::from_data(
                &memory_allocator,
                BufferAllocateInfo {
                    buffer_usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                0u32,
            )
            .unwrap();
            let mut buf_builder = sync.bind_vertex_buffers();
            buf_builder.add(buf.into_bytes());
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

            let cb_allocator =
                StandardCommandBufferAllocator::new(device.clone(), Default::default());
            let builder_alloc = cb_allocator
                .allocate(queue.queue_family_index(), CommandBufferLevel::Primary, 1)
                .unwrap()
                .next()
                .unwrap();
            let mut sync = SyncCommandBufferBuilder::new(
                builder_alloc.inner(),
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
