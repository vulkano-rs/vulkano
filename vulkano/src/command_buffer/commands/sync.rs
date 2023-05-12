// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    command_buffer::{allocator::CommandBufferAllocator, sys::UnsafeCommandBufferBuilder},
    device::DeviceOwned,
    sync::{
        event::Event, BufferMemoryBarrier, DependencyFlags, DependencyInfo, ImageMemoryBarrier,
        MemoryBarrier, PipelineStages,
    },
    Version, VulkanObject,
};
use smallvec::SmallVec;
use std::{ptr, sync::Arc};

impl<A> UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    #[inline]
    pub unsafe fn pipeline_barrier(&mut self, dependency_info: DependencyInfo) -> &mut Self {
        if dependency_info.is_empty() {
            return self;
        }

        let DependencyInfo {
            mut dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne: _,
        } = dependency_info;

        // TODO: Is this needed?
        dependency_flags |= DependencyFlags::BY_REGION;

        if self.device().enabled_features().synchronization2 {
            let memory_barriers_vk: SmallVec<[_; 2]> = memory_barriers
                .iter()
                .map(|barrier| {
                    let &MemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        _ne: _,
                    } = barrier;

                    ash::vk::MemoryBarrier2 {
                        src_stage_mask: src_stages.into(),
                        src_access_mask: src_access.into(),
                        dst_stage_mask: dst_stages.into(),
                        dst_access_mask: dst_access.into(),
                        ..Default::default()
                    }
                })
                .collect();

            let buffer_memory_barriers_vk: SmallVec<[_; 8]> = buffer_memory_barriers
                .iter()
                .map(|barrier| {
                    let &BufferMemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        queue_family_ownership_transfer,
                        ref buffer,
                        ref range,
                        _ne: _,
                    } = barrier;

                    let (src_queue_family_index, dst_queue_family_index) =
                        queue_family_ownership_transfer.map_or(
                            (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED),
                            Into::into,
                        );

                    ash::vk::BufferMemoryBarrier2 {
                        src_stage_mask: src_stages.into(),
                        src_access_mask: src_access.into(),
                        dst_stage_mask: dst_stages.into(),
                        dst_access_mask: dst_access.into(),
                        src_queue_family_index,
                        dst_queue_family_index,
                        buffer: buffer.handle(),
                        offset: range.start,
                        size: range.end - range.start,
                        ..Default::default()
                    }
                })
                .collect();

            let image_memory_barriers_vk: SmallVec<[_; 8]> = image_memory_barriers
                .iter()
                .map(|barrier| {
                    let &ImageMemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        old_layout,
                        new_layout,
                        queue_family_ownership_transfer,
                        ref image,
                        ref subresource_range,
                        _ne: _,
                    } = barrier;

                    let (src_queue_family_index, dst_queue_family_index) =
                        queue_family_ownership_transfer.map_or(
                            (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED),
                            Into::into,
                        );

                    ash::vk::ImageMemoryBarrier2 {
                        src_stage_mask: src_stages.into(),
                        src_access_mask: src_access.into(),
                        dst_stage_mask: dst_stages.into(),
                        dst_access_mask: dst_access.into(),
                        old_layout: old_layout.into(),
                        new_layout: new_layout.into(),
                        src_queue_family_index,
                        dst_queue_family_index,
                        image: image.handle(),
                        subresource_range: subresource_range.clone().into(),
                        ..Default::default()
                    }
                })
                .collect();

            let dependency_info_vk = ash::vk::DependencyInfo {
                dependency_flags: dependency_flags.into(),
                memory_barrier_count: memory_barriers_vk.len() as u32,
                p_memory_barriers: memory_barriers_vk.as_ptr(),
                buffer_memory_barrier_count: buffer_memory_barriers_vk.len() as u32,
                p_buffer_memory_barriers: buffer_memory_barriers_vk.as_ptr(),
                image_memory_barrier_count: image_memory_barriers_vk.len() as u32,
                p_image_memory_barriers: image_memory_barriers_vk.as_ptr(),
                ..Default::default()
            };

            let fns = self.device().fns();

            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_pipeline_barrier2)(self.handle(), &dependency_info_vk);
            } else {
                (fns.khr_synchronization2.cmd_pipeline_barrier2_khr)(
                    self.handle(),
                    &dependency_info_vk,
                );
            }
        } else {
            let mut src_stage_mask = ash::vk::PipelineStageFlags::empty();
            let mut dst_stage_mask = ash::vk::PipelineStageFlags::empty();

            let memory_barriers_vk: SmallVec<[_; 2]> = memory_barriers
                .iter()
                .map(|barrier| {
                    let &MemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        _ne: _,
                    } = barrier;

                    src_stage_mask |= src_stages.into();
                    dst_stage_mask |= dst_stages.into();

                    ash::vk::MemoryBarrier {
                        src_access_mask: src_access.into(),
                        dst_access_mask: dst_access.into(),
                        ..Default::default()
                    }
                })
                .collect();

            let buffer_memory_barriers_vk: SmallVec<[_; 8]> = buffer_memory_barriers
                .iter()
                .map(|barrier| {
                    let &BufferMemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        queue_family_ownership_transfer,
                        ref buffer,
                        ref range,
                        _ne: _,
                    } = barrier;

                    src_stage_mask |= src_stages.into();
                    dst_stage_mask |= dst_stages.into();

                    let (src_queue_family_index, dst_queue_family_index) =
                        queue_family_ownership_transfer.map_or(
                            (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED),
                            Into::into,
                        );

                    ash::vk::BufferMemoryBarrier {
                        src_access_mask: src_access.into(),
                        dst_access_mask: dst_access.into(),
                        src_queue_family_index,
                        dst_queue_family_index,
                        buffer: buffer.handle(),
                        offset: range.start,
                        size: range.end - range.start,
                        ..Default::default()
                    }
                })
                .collect();

            let image_memory_barriers_vk: SmallVec<[_; 8]> = image_memory_barriers
                .iter()
                .map(|barrier| {
                    let &ImageMemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        old_layout,
                        new_layout,
                        queue_family_ownership_transfer,
                        ref image,
                        ref subresource_range,
                        _ne: _,
                    } = barrier;

                    src_stage_mask |= src_stages.into();
                    dst_stage_mask |= dst_stages.into();

                    let (src_queue_family_index, dst_queue_family_index) =
                        queue_family_ownership_transfer.map_or(
                            (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED),
                            Into::into,
                        );

                    ash::vk::ImageMemoryBarrier {
                        src_access_mask: src_access.into(),
                        dst_access_mask: dst_access.into(),
                        old_layout: old_layout.into(),
                        new_layout: new_layout.into(),
                        src_queue_family_index,
                        dst_queue_family_index,
                        image: image.handle(),
                        subresource_range: subresource_range.clone().into(),
                        ..Default::default()
                    }
                })
                .collect();

            if src_stage_mask.is_empty() {
                // "VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is [...] equivalent to
                // VK_PIPELINE_STAGE_2_NONE in the first scope."
                src_stage_mask |= ash::vk::PipelineStageFlags::TOP_OF_PIPE;
            }

            if dst_stage_mask.is_empty() {
                // "VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT is [...] equivalent to
                // VK_PIPELINE_STAGE_2_NONE in the second scope."
                dst_stage_mask |= ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE;
            }

            let fns = self.device().fns();
            (fns.v1_0.cmd_pipeline_barrier)(
                self.handle(),
                src_stage_mask,
                dst_stage_mask,
                dependency_flags.into(),
                memory_barriers_vk.len() as u32,
                memory_barriers_vk.as_ptr(),
                buffer_memory_barriers_vk.len() as u32,
                buffer_memory_barriers_vk.as_ptr(),
                image_memory_barriers_vk.len() as u32,
                image_memory_barriers_vk.as_ptr(),
            );
        }

        self.keep_alive_objects.extend(
            (buffer_memory_barriers
                .into_iter()
                .map(|barrier| Box::new(barrier.buffer) as _))
            .chain(
                image_memory_barriers
                    .into_iter()
                    .map(|barrier| Box::new(barrier.image) as _),
            ),
        );

        self
    }

    /// Calls `vkCmdSetEvent` on the builder.
    #[inline]
    pub unsafe fn set_event(
        &mut self,
        event: Arc<Event>,
        dependency_info: DependencyInfo,
    ) -> &mut Self {
        let DependencyInfo {
            mut dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne: _,
        } = dependency_info;

        // TODO: Is this needed?
        dependency_flags |= DependencyFlags::BY_REGION;

        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            let memory_barriers_vk: SmallVec<[_; 2]> = memory_barriers
                .iter()
                .map(|barrier| {
                    let &MemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        _ne: _,
                    } = barrier;

                    ash::vk::MemoryBarrier2 {
                        src_stage_mask: src_stages.into(),
                        src_access_mask: src_access.into(),
                        dst_stage_mask: dst_stages.into(),
                        dst_access_mask: dst_access.into(),
                        ..Default::default()
                    }
                })
                .collect();

            let buffer_memory_barriers_vk: SmallVec<[_; 8]> = buffer_memory_barriers
                .iter()
                .map(|barrier| {
                    let &BufferMemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        queue_family_ownership_transfer,
                        ref buffer,
                        ref range,
                        _ne: _,
                    } = barrier;

                    let (src_queue_family_index, dst_queue_family_index) =
                        queue_family_ownership_transfer.map_or(
                            (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED),
                            Into::into,
                        );

                    ash::vk::BufferMemoryBarrier2 {
                        src_stage_mask: src_stages.into(),
                        src_access_mask: src_access.into(),
                        dst_stage_mask: dst_stages.into(),
                        dst_access_mask: dst_access.into(),
                        src_queue_family_index,
                        dst_queue_family_index,
                        buffer: buffer.handle(),
                        offset: range.start,
                        size: range.end - range.start,
                        ..Default::default()
                    }
                })
                .collect();

            let image_memory_barriers_vk: SmallVec<[_; 8]> = image_memory_barriers
                .iter()
                .map(|barrier| {
                    let &ImageMemoryBarrier {
                        src_stages,
                        src_access,
                        dst_stages,
                        dst_access,
                        old_layout,
                        new_layout,
                        queue_family_ownership_transfer,
                        ref image,
                        ref subresource_range,
                        _ne: _,
                    } = barrier;

                    let (src_queue_family_index, dst_queue_family_index) =
                        queue_family_ownership_transfer.map_or(
                            (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED),
                            Into::into,
                        );

                    ash::vk::ImageMemoryBarrier2 {
                        src_stage_mask: src_stages.into(),
                        src_access_mask: src_access.into(),
                        dst_stage_mask: dst_stages.into(),
                        dst_access_mask: dst_access.into(),
                        old_layout: old_layout.into(),
                        new_layout: new_layout.into(),
                        src_queue_family_index,
                        dst_queue_family_index,
                        image: image.handle(),
                        subresource_range: subresource_range.clone().into(),
                        ..Default::default()
                    }
                })
                .collect();

            let dependency_info_vk = ash::vk::DependencyInfo {
                dependency_flags: dependency_flags.into(),
                memory_barrier_count: memory_barriers_vk.len() as u32,
                p_memory_barriers: memory_barriers_vk.as_ptr(),
                buffer_memory_barrier_count: buffer_memory_barriers_vk.len() as u32,
                p_buffer_memory_barriers: buffer_memory_barriers_vk.as_ptr(),
                image_memory_barrier_count: image_memory_barriers_vk.len() as u32,
                p_image_memory_barriers: image_memory_barriers_vk.as_ptr(),
                ..Default::default()
            };

            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_set_event2)(self.handle(), event.handle(), &dependency_info_vk);
            } else {
                (fns.khr_synchronization2.cmd_set_event2_khr)(
                    self.handle(),
                    event.handle(),
                    &dependency_info_vk,
                );
            }
        } else {
            // The original function only takes a source stage mask; the rest of the info is
            // provided with `wait_events` instead. Therefore, we condense the source stages
            // here and ignore the rest.

            let mut stage_mask = ash::vk::PipelineStageFlags::empty();

            for barrier in memory_barriers {
                stage_mask |= barrier.src_stages.into();
            }

            for barrier in buffer_memory_barriers {
                stage_mask |= barrier.src_stages.into();
            }

            for barrier in image_memory_barriers {
                stage_mask |= barrier.src_stages.into();
            }

            if stage_mask.is_empty() {
                // "VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is [...] equivalent to
                // VK_PIPELINE_STAGE_2_NONE in the first scope."
                stage_mask |= ash::vk::PipelineStageFlags::TOP_OF_PIPE;
            }

            (fns.v1_0.cmd_set_event)(self.handle(), event.handle(), stage_mask);
        }

        self.keep_alive_objects.push(Box::new(event));

        self
    }

    /// Calls `vkCmdWaitEvents` on the builder.
    pub unsafe fn wait_events(
        &mut self,
        events: impl IntoIterator<Item = (Arc<Event>, DependencyInfo)>,
    ) -> &mut Self {
        let events: SmallVec<[(Arc<Event>, DependencyInfo); 4]> = events.into_iter().collect();
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            struct PerDependencyInfo {
                memory_barriers_vk: SmallVec<[ash::vk::MemoryBarrier2; 2]>,
                buffer_memory_barriers_vk: SmallVec<[ash::vk::BufferMemoryBarrier2; 8]>,
                image_memory_barriers_vk: SmallVec<[ash::vk::ImageMemoryBarrier2; 8]>,
            }

            let mut events_vk: SmallVec<[_; 4]> = SmallVec::new();
            let mut dependency_infos_vk: SmallVec<[_; 4]> = SmallVec::new();
            let mut per_dependency_info_vk: SmallVec<[_; 4]> = SmallVec::new();

            for (event, dependency_info) in &events {
                let &DependencyInfo {
                    mut dependency_flags,
                    ref memory_barriers,
                    ref buffer_memory_barriers,
                    ref image_memory_barriers,
                    _ne: _,
                } = dependency_info;

                // TODO: Is this needed?
                dependency_flags |= DependencyFlags::BY_REGION;

                let memory_barriers_vk: SmallVec<[_; 2]> = memory_barriers
                    .iter()
                    .map(|barrier| {
                        let &MemoryBarrier {
                            src_stages,
                            src_access,
                            dst_stages,
                            dst_access,
                            _ne: _,
                        } = barrier;

                        ash::vk::MemoryBarrier2 {
                            src_stage_mask: src_stages.into(),
                            src_access_mask: src_access.into(),
                            dst_stage_mask: dst_stages.into(),
                            dst_access_mask: dst_access.into(),
                            ..Default::default()
                        }
                    })
                    .collect();

                let buffer_memory_barriers_vk: SmallVec<[_; 8]> = buffer_memory_barriers
                    .iter()
                    .map(|barrier| {
                        let &BufferMemoryBarrier {
                            src_stages,
                            src_access,
                            dst_stages,
                            dst_access,
                            queue_family_ownership_transfer,
                            ref buffer,
                            ref range,
                            _ne: _,
                        } = barrier;

                        let (src_queue_family_index, dst_queue_family_index) =
                            queue_family_ownership_transfer.map_or(
                                (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED),
                                Into::into,
                            );

                        ash::vk::BufferMemoryBarrier2 {
                            src_stage_mask: src_stages.into(),
                            src_access_mask: src_access.into(),
                            dst_stage_mask: dst_stages.into(),
                            dst_access_mask: dst_access.into(),
                            src_queue_family_index,
                            dst_queue_family_index,
                            buffer: buffer.handle(),
                            offset: range.start,
                            size: range.end - range.start,
                            ..Default::default()
                        }
                    })
                    .collect();

                let image_memory_barriers_vk: SmallVec<[_; 8]> = image_memory_barriers
                    .iter()
                    .map(|barrier| {
                        let &ImageMemoryBarrier {
                            src_stages,
                            src_access,
                            dst_stages,
                            dst_access,
                            old_layout,
                            new_layout,
                            queue_family_ownership_transfer,
                            ref image,
                            ref subresource_range,
                            _ne: _,
                        } = barrier;

                        let (src_queue_family_index, dst_queue_family_index) =
                            queue_family_ownership_transfer.map_or(
                                (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED),
                                Into::into,
                            );

                        ash::vk::ImageMemoryBarrier2 {
                            src_stage_mask: src_stages.into(),
                            src_access_mask: src_access.into(),
                            dst_stage_mask: dst_stages.into(),
                            dst_access_mask: dst_access.into(),
                            old_layout: old_layout.into(),
                            new_layout: new_layout.into(),
                            src_queue_family_index,
                            dst_queue_family_index,
                            image: image.handle(),
                            subresource_range: subresource_range.clone().into(),
                            ..Default::default()
                        }
                    })
                    .collect();

                events_vk.push(event.handle());
                dependency_infos_vk.push(ash::vk::DependencyInfo {
                    dependency_flags: dependency_flags.into(),
                    memory_barrier_count: 0,
                    p_memory_barriers: ptr::null(),
                    buffer_memory_barrier_count: 0,
                    p_buffer_memory_barriers: ptr::null(),
                    image_memory_barrier_count: 0,
                    p_image_memory_barriers: ptr::null(),
                    ..Default::default()
                });
                per_dependency_info_vk.push(PerDependencyInfo {
                    memory_barriers_vk,
                    buffer_memory_barriers_vk,
                    image_memory_barriers_vk,
                });
            }

            for (
                dependency_info_vk,
                PerDependencyInfo {
                    memory_barriers_vk,
                    buffer_memory_barriers_vk,
                    image_memory_barriers_vk,
                },
            ) in (dependency_infos_vk.iter_mut()).zip(per_dependency_info_vk.iter_mut())
            {
                *dependency_info_vk = ash::vk::DependencyInfo {
                    memory_barrier_count: memory_barriers_vk.len() as u32,
                    p_memory_barriers: memory_barriers_vk.as_ptr(),
                    buffer_memory_barrier_count: buffer_memory_barriers_vk.len() as u32,
                    p_buffer_memory_barriers: buffer_memory_barriers_vk.as_ptr(),
                    image_memory_barrier_count: image_memory_barriers_vk.len() as u32,
                    p_image_memory_barriers: image_memory_barriers_vk.as_ptr(),
                    ..*dependency_info_vk
                }
            }

            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_wait_events2)(
                    self.handle(),
                    events_vk.len() as u32,
                    events_vk.as_ptr(),
                    dependency_infos_vk.as_ptr(),
                );
            } else {
                (fns.khr_synchronization2.cmd_wait_events2_khr)(
                    self.handle(),
                    events_vk.len() as u32,
                    events_vk.as_ptr(),
                    dependency_infos_vk.as_ptr(),
                );
            }
        } else {
            // With the original function, you can only specify a single dependency info for all
            // events at once, rather than separately for each event. Therefore, to achieve the
            // same behaviour as the "2" function, we split it up into multiple Vulkan API calls,
            // one per event.

            for (event, dependency_info) in &events {
                let events_vk = [event.handle()];

                let &DependencyInfo {
                    dependency_flags: _,
                    ref memory_barriers,
                    ref buffer_memory_barriers,
                    ref image_memory_barriers,
                    _ne: _,
                } = dependency_info;

                let mut src_stage_mask = ash::vk::PipelineStageFlags::empty();
                let mut dst_stage_mask = ash::vk::PipelineStageFlags::empty();

                let memory_barriers_vk: SmallVec<[_; 2]> = memory_barriers
                    .iter()
                    .map(|barrier| {
                        let &MemoryBarrier {
                            src_stages,
                            src_access,
                            dst_stages,
                            dst_access,
                            _ne: _,
                        } = barrier;

                        src_stage_mask |= src_stages.into();
                        dst_stage_mask |= dst_stages.into();

                        ash::vk::MemoryBarrier {
                            src_access_mask: src_access.into(),
                            dst_access_mask: dst_access.into(),
                            ..Default::default()
                        }
                    })
                    .collect();

                let buffer_memory_barriers_vk: SmallVec<[_; 8]> = buffer_memory_barriers
                    .iter()
                    .map(|barrier| {
                        let &BufferMemoryBarrier {
                            src_stages,
                            src_access,
                            dst_stages,
                            dst_access,
                            queue_family_ownership_transfer,
                            ref buffer,
                            ref range,
                            _ne: _,
                        } = barrier;

                        src_stage_mask |= src_stages.into();
                        dst_stage_mask |= dst_stages.into();

                        let (src_queue_family_index, dst_queue_family_index) =
                            queue_family_ownership_transfer.map_or(
                                (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED),
                                Into::into,
                            );

                        ash::vk::BufferMemoryBarrier {
                            src_access_mask: src_access.into(),
                            dst_access_mask: dst_access.into(),
                            src_queue_family_index,
                            dst_queue_family_index,
                            buffer: buffer.handle(),
                            offset: range.start,
                            size: range.end - range.start,
                            ..Default::default()
                        }
                    })
                    .collect();

                let image_memory_barriers_vk: SmallVec<[_; 8]> = image_memory_barriers
                    .iter()
                    .map(|barrier| {
                        let &ImageMemoryBarrier {
                            src_stages,
                            src_access,
                            dst_stages,
                            dst_access,
                            old_layout,
                            new_layout,
                            queue_family_ownership_transfer,
                            ref image,
                            ref subresource_range,
                            _ne: _,
                        } = barrier;

                        src_stage_mask |= src_stages.into();
                        dst_stage_mask |= dst_stages.into();

                        let (src_queue_family_index, dst_queue_family_index) =
                            queue_family_ownership_transfer.map_or(
                                (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED),
                                Into::into,
                            );

                        ash::vk::ImageMemoryBarrier {
                            src_access_mask: src_access.into(),
                            dst_access_mask: dst_access.into(),
                            old_layout: old_layout.into(),
                            new_layout: new_layout.into(),
                            src_queue_family_index,
                            dst_queue_family_index,
                            image: image.handle(),
                            subresource_range: subresource_range.clone().into(),
                            ..Default::default()
                        }
                    })
                    .collect();

                if src_stage_mask.is_empty() {
                    // "VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is [...] equivalent to
                    // VK_PIPELINE_STAGE_2_NONE in the first scope."
                    src_stage_mask |= ash::vk::PipelineStageFlags::TOP_OF_PIPE;
                }

                if dst_stage_mask.is_empty() {
                    // "VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT is [...] equivalent to
                    // VK_PIPELINE_STAGE_2_NONE in the second scope."
                    dst_stage_mask |= ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                }

                (fns.v1_0.cmd_wait_events)(
                    self.handle(),
                    1,
                    events_vk.as_ptr(),
                    src_stage_mask,
                    dst_stage_mask,
                    memory_barriers_vk.len() as u32,
                    memory_barriers_vk.as_ptr(),
                    buffer_memory_barriers_vk.len() as u32,
                    buffer_memory_barriers_vk.as_ptr(),
                    image_memory_barriers_vk.len() as u32,
                    image_memory_barriers_vk.as_ptr(),
                );
            }
        }

        self.keep_alive_objects
            .extend(events.into_iter().map(|(event, _)| Box::new(event) as _));

        self
    }

    /// Calls `vkCmdResetEvent` on the builder.
    #[inline]
    pub unsafe fn reset_event(&mut self, event: Arc<Event>, stages: PipelineStages) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_reset_event2)(self.handle(), event.handle(), stages.into());
            } else {
                (fns.khr_synchronization2.cmd_reset_event2_khr)(
                    self.handle(),
                    event.handle(),
                    stages.into(),
                );
            }
        } else {
            (fns.v1_0.cmd_reset_event)(self.handle(), event.handle(), stages.into());
        }

        self.keep_alive_objects.push(Box::new(event));

        self
    }
}
