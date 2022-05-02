// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    command_buffer::{
        synced::{Command, SyncCommandBufferBuilder},
        sys::UnsafeCommandBufferBuilder,
    },
    image::ImageLayout,
    sync::{
        BufferMemoryBarrier, DependencyInfo, Event, ImageMemoryBarrier, MemoryBarrier,
        PipelineStages,
    },
    Version, VulkanObject,
};
use smallvec::SmallVec;
use std::sync::Arc;

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdSetEvent` on the builder.
    #[inline]
    pub unsafe fn set_event(&mut self, event: Arc<Event>, stages: PipelineStages) {
        struct Cmd {
            event: Arc<Event>,
            stages: PipelineStages,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_event"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_event(&self.event, self.stages);
            }
        }

        self.commands.push(Box::new(Cmd { event, stages }));
    }

    /// Calls `vkCmdResetEvent` on the builder.
    #[inline]
    pub unsafe fn reset_event(&mut self, event: Arc<Event>, stages: PipelineStages) {
        struct Cmd {
            event: Arc<Event>,
            stages: PipelineStages,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "reset_event"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.reset_event(&self.event, self.stages);
            }
        }

        self.commands.push(Box::new(Cmd { event, stages }));
    }
}

impl UnsafeCommandBufferBuilder {
    #[inline]
    pub unsafe fn pipeline_barrier(&mut self, dependency_info: &DependencyInfo) {
        if dependency_info.is_empty() {
            return;
        }

        let DependencyInfo {
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne: _,
        } = dependency_info;

        let dependency_flags = ash::vk::DependencyFlags::BY_REGION;

        if self.device.enabled_features().synchronization2 {
            let memory_barriers: SmallVec<[_; 2]> = memory_barriers
                .into_iter()
                .map(|barrier| {
                    let &MemoryBarrier {
                        source_stages,
                        source_access,
                        destination_stages,
                        destination_access,
                        _ne: _,
                    } = barrier;

                    debug_assert!(source_stages.supported_access().contains(&source_access));
                    debug_assert!(destination_stages
                        .supported_access()
                        .contains(&destination_access));

                    ash::vk::MemoryBarrier2 {
                        src_stage_mask: source_stages.into(),
                        src_access_mask: source_access.into(),
                        dst_stage_mask: destination_stages.into(),
                        dst_access_mask: destination_access.into(),
                        ..Default::default()
                    }
                })
                .collect();

            let buffer_memory_barriers: SmallVec<[_; 8]> = buffer_memory_barriers
                .into_iter()
                .map(|barrier| {
                    let &BufferMemoryBarrier {
                        source_stages,
                        source_access,
                        destination_stages,
                        destination_access,
                        queue_family_transfer,
                        ref buffer,
                        ref range,
                        _ne: _,
                    } = barrier;

                    debug_assert!(source_stages.supported_access().contains(&source_access));
                    debug_assert!(destination_stages
                        .supported_access()
                        .contains(&destination_access));
                    debug_assert!(!range.is_empty());
                    debug_assert!(range.end <= buffer.size());

                    ash::vk::BufferMemoryBarrier2 {
                        src_stage_mask: source_stages.into(),
                        src_access_mask: source_access.into(),
                        dst_stage_mask: destination_stages.into(),
                        dst_access_mask: destination_access.into(),
                        src_queue_family_index: queue_family_transfer
                            .map_or(ash::vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.source_index
                            }),
                        dst_queue_family_index: queue_family_transfer
                            .map_or(ash::vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.destination_index
                            }),
                        buffer: buffer.internal_object(),
                        offset: range.start,
                        size: range.end - range.start,
                        ..Default::default()
                    }
                })
                .collect();

            let image_memory_barriers: SmallVec<[_; 8]> = image_memory_barriers
                .into_iter()
                .map(|barrier| {
                    let &ImageMemoryBarrier {
                        source_stages,
                        source_access,
                        destination_stages,
                        destination_access,
                        old_layout,
                        new_layout,
                        queue_family_transfer,
                        ref image,
                        ref subresource_range,
                        _ne: _,
                    } = barrier;

                    debug_assert!(source_stages.supported_access().contains(&source_access));
                    debug_assert!(destination_stages
                        .supported_access()
                        .contains(&destination_access));
                    debug_assert!(!matches!(
                        new_layout,
                        ImageLayout::Undefined | ImageLayout::Preinitialized
                    ));
                    debug_assert!(image
                        .format()
                        .unwrap()
                        .aspects()
                        .contains(&subresource_range.aspects));
                    debug_assert!(!subresource_range.mip_levels.is_empty());
                    debug_assert!(subresource_range.mip_levels.end <= image.mip_levels());
                    debug_assert!(!subresource_range.array_layers.is_empty());
                    debug_assert!(
                        subresource_range.array_layers.end <= image.dimensions().array_layers()
                    );

                    ash::vk::ImageMemoryBarrier2 {
                        src_stage_mask: source_stages.into(),
                        src_access_mask: source_access.into(),
                        dst_stage_mask: destination_stages.into(),
                        dst_access_mask: destination_access.into(),
                        old_layout: old_layout.into(),
                        new_layout: new_layout.into(),
                        src_queue_family_index: queue_family_transfer
                            .map_or(ash::vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.source_index
                            }),
                        dst_queue_family_index: queue_family_transfer
                            .map_or(ash::vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.destination_index
                            }),
                        image: image.internal_object(),
                        subresource_range: subresource_range.clone().into(),
                        ..Default::default()
                    }
                })
                .collect();

            let dependency_info = ash::vk::DependencyInfo {
                dependency_flags,
                memory_barrier_count: memory_barriers.len() as u32,
                p_memory_barriers: memory_barriers.as_ptr(),
                buffer_memory_barrier_count: buffer_memory_barriers.len() as u32,
                p_buffer_memory_barriers: buffer_memory_barriers.as_ptr(),
                image_memory_barrier_count: image_memory_barriers.len() as u32,
                p_image_memory_barriers: image_memory_barriers.as_ptr(),
                ..Default::default()
            };

            let fns = self.device.fns();

            if self.device.api_version() >= Version::V1_3 {
                fns.v1_3
                    .cmd_pipeline_barrier2(self.handle, &dependency_info);
            } else {
                fns.khr_synchronization2
                    .cmd_pipeline_barrier2_khr(self.handle, &dependency_info);
            }
        } else {
            let mut src_stage_mask = ash::vk::PipelineStageFlags::empty();
            let mut dst_stage_mask = ash::vk::PipelineStageFlags::empty();

            let memory_barriers: SmallVec<[_; 2]> = memory_barriers
                .into_iter()
                .map(|barrier| {
                    let &MemoryBarrier {
                        source_stages,
                        source_access,
                        destination_stages,
                        destination_access,
                        _ne: _,
                    } = barrier;

                    debug_assert!(source_stages.supported_access().contains(&source_access));
                    debug_assert!(destination_stages
                        .supported_access()
                        .contains(&destination_access));

                    src_stage_mask |= source_stages.into();
                    dst_stage_mask |= destination_stages.into();

                    ash::vk::MemoryBarrier {
                        src_access_mask: source_access.into(),
                        dst_access_mask: destination_access.into(),
                        ..Default::default()
                    }
                })
                .collect();

            let buffer_memory_barriers: SmallVec<[_; 8]> = buffer_memory_barriers
                .into_iter()
                .map(|barrier| {
                    let &BufferMemoryBarrier {
                        source_stages,
                        source_access,
                        destination_stages,
                        destination_access,
                        queue_family_transfer,
                        ref buffer,
                        ref range,
                        _ne: _,
                    } = barrier;

                    debug_assert!(source_stages.supported_access().contains(&source_access));
                    debug_assert!(destination_stages
                        .supported_access()
                        .contains(&destination_access));
                    debug_assert!(!range.is_empty());
                    debug_assert!(range.end <= buffer.size());

                    src_stage_mask |= source_stages.into();
                    dst_stage_mask |= destination_stages.into();

                    ash::vk::BufferMemoryBarrier {
                        src_access_mask: source_access.into(),
                        dst_access_mask: destination_access.into(),
                        src_queue_family_index: queue_family_transfer
                            .map_or(ash::vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.source_index
                            }),
                        dst_queue_family_index: queue_family_transfer
                            .map_or(ash::vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.destination_index
                            }),
                        buffer: buffer.internal_object(),
                        offset: range.start,
                        size: range.end - range.start,
                        ..Default::default()
                    }
                })
                .collect();

            let image_memory_barriers: SmallVec<[_; 8]> = image_memory_barriers
                .into_iter()
                .map(|barrier| {
                    let &ImageMemoryBarrier {
                        source_stages,
                        source_access,
                        destination_stages,
                        destination_access,
                        old_layout,
                        new_layout,
                        queue_family_transfer,
                        ref image,
                        ref subresource_range,
                        _ne: _,
                    } = barrier;

                    debug_assert!(source_stages.supported_access().contains(&source_access));
                    debug_assert!(destination_stages
                        .supported_access()
                        .contains(&destination_access));
                    debug_assert!(!matches!(
                        new_layout,
                        ImageLayout::Undefined | ImageLayout::Preinitialized
                    ));
                    debug_assert!(image
                        .format()
                        .unwrap()
                        .aspects()
                        .contains(&subresource_range.aspects));
                    debug_assert!(!subresource_range.mip_levels.is_empty());
                    debug_assert!(subresource_range.mip_levels.end <= image.mip_levels());
                    debug_assert!(!subresource_range.array_layers.is_empty());
                    debug_assert!(
                        subresource_range.array_layers.end <= image.dimensions().array_layers()
                    );

                    src_stage_mask |= source_stages.into();
                    dst_stage_mask |= destination_stages.into();

                    ash::vk::ImageMemoryBarrier {
                        src_access_mask: source_access.into(),
                        dst_access_mask: destination_access.into(),
                        old_layout: old_layout.into(),
                        new_layout: new_layout.into(),
                        src_queue_family_index: queue_family_transfer
                            .map_or(ash::vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.source_index
                            }),
                        dst_queue_family_index: queue_family_transfer
                            .map_or(ash::vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.destination_index
                            }),
                        image: image.internal_object(),
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

            let fns = self.device.fns();
            fns.v1_0.cmd_pipeline_barrier(
                self.handle,
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                memory_barriers.len() as u32,
                memory_barriers.as_ptr(),
                buffer_memory_barriers.len() as u32,
                buffer_memory_barriers.as_ptr(),
                image_memory_barriers.len() as u32,
                image_memory_barriers.as_ptr(),
            );
        }
    }

    /// Calls `vkCmdSetEvent` on the builder.
    #[inline]
    pub unsafe fn set_event(&mut self, event: &Event, stages: PipelineStages) {
        debug_assert!(!stages.host);
        debug_assert_ne!(stages, PipelineStages::none());
        let fns = self.device.fns();
        fns.v1_0
            .cmd_set_event(self.handle, event.internal_object(), stages.into());
    }

    /// Calls `vkCmdResetEvent` on the builder.
    #[inline]
    pub unsafe fn reset_event(&mut self, event: &Event, stages: PipelineStages) {
        let fns = self.device.fns();

        debug_assert!(!stages.host);
        debug_assert_ne!(stages, PipelineStages::none());

        fns.v1_0
            .cmd_reset_event(self.handle, event.internal_object(), stages.into());
    }
}
