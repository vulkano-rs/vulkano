// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{CommandBufferBuilder, RenderPassStateType};
use crate::{
    command_buffer::allocator::CommandBufferAllocator,
    device::{DeviceOwned, QueueFlags},
    image::{ImageAspects, ImageCreateFlags, ImageLayout, ImageUsage},
    sync::{
        event::Event, AccessFlags, BufferMemoryBarrier, DependencyFlags, DependencyInfo,
        ImageMemoryBarrier, MemoryBarrier, PipelineStages, QueueFamilyOwnershipTransfer, Sharing,
    },
    DeviceSize, RequirementNotMet, RequiresOneOf, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    cmp::max,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    ptr,
    sync::Arc,
};

impl<L, A> CommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Inserts a memory dependency into the queue.
    ///
    /// A pipeline barrier is the primary means of synchronizing work within a single queue.
    /// Without a pipeline barrier, all commands in a queue could complete out of order, and there
    /// would be no guarantee about the ordering of memory accesses. When a dependency is inserted,
    /// a chosen subset of operations after the barrier (the destination synchronization scope)
    /// depend on, and therefore must wait for the completion of, a chosen subset operations before
    /// barrier (the source synchronization scope).
    ///
    /// When the queue has to wait, it may not be doing useful work, so barriers should be kept to
    /// the minimum needed to ensure proper functioning. Overly broad barriers, specifying
    /// a larger class of operations than needed (especially [`ALL_COMMANDS`]), can slow down the
    /// queue and make it work less efficiently.
    ///
    /// In addition to ensuring correct operation, pipeline barriers are also used to transition
    /// images (or parts of images) from one [image layout] to another.
    ///
    /// # Safety
    ///
    /// - All images that are accessed by the command must be in the expected image layout.
    /// - For each element of `dependency_info.image_memory_barriers` that contains an image layout
    ///   transition, which is a write operation, the barrier must be defined appropriately to
    ///   ensure no memory access hazard occurs.
    ///
    /// [`ALL_COMMANDS`]: PipelineStages::ALL_COMMANDS
    /// [image layout]: ImageLayout
    #[inline]
    pub unsafe fn pipeline_barrier(
        &mut self,
        dependency_info: DependencyInfo,
    ) -> Result<&mut Self, SynchronizationError> {
        self.validate_pipeline_barrier(&dependency_info)?;

        unsafe { Ok(self.pipeline_barrier_unchecked(dependency_info)) }
    }

    fn validate_pipeline_barrier(
        &self,
        dependency_info: &DependencyInfo,
    ) -> Result<(), SynchronizationError> {
        let device = self.device();
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdPipelineBarrier2-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::TRANSFER
                | QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(SynchronizationError::NotSupportedByQueueFamily);
        }

        let &DependencyInfo {
            dependency_flags,
            ref memory_barriers,
            ref buffer_memory_barriers,
            ref image_memory_barriers,
            _ne: _,
        } = dependency_info;

        // VUID-VkDependencyInfo-dependencyFlags-parameter
        dependency_flags.validate_device(device)?;

        let check_stages_access = |ty: char,
                                   barrier_index: usize,
                                   src_stages: PipelineStages,
                                   src_access: AccessFlags,
                                   dst_stages: PipelineStages,
                                   dst_access: AccessFlags|
         -> Result<(), SynchronizationError> {
            for (stages, access) in [(src_stages, src_access), (dst_stages, dst_access)] {
                // VUID-vkCmdPipelineBarrier2-synchronization2-03848
                if !device.enabled_features().synchronization2 {
                    if stages.is_2() {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                                `dependency_info.buffer_memory_barriers` or \
                                `dependency_info.image_memory_barriers` has an element where \
                                `src_stages` or `dst_stages` contains flags from \
                                `VkPipelineStageFlagBits2`",
                            requires_one_of: RequiresOneOf {
                                features: &["synchronization2"],
                                ..Default::default()
                            },
                        });
                    }

                    if access.is_2() {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                                `dependency_info.buffer_memory_barriers` or \
                                `dependency_info.image_memory_barriers` has an element where \
                                `src_access` or `dst_access` contains flags from \
                                `VkAccessFlagBits2`",
                            requires_one_of: RequiresOneOf {
                                features: &["synchronization2"],
                                ..Default::default()
                            },
                        });
                    }
                }

                // VUID-VkMemoryBarrier2-srcStageMask-parameter
                // VUID-VkMemoryBarrier2-dstStageMask-parameter
                // VUID-VkBufferMemoryBarrier2-srcStageMask-parameter
                // VUID-VkBufferMemoryBarrier2-dstStageMask-parameter
                // VUID-VkImageMemoryBarrier2-srcStageMask-parameter
                // VUID-VkImageMemoryBarrier2-dstStageMask-parameter
                stages.validate_device(device)?;

                // VUID-VkMemoryBarrier2-srcAccessMask-parameter
                // VUID-VkMemoryBarrier2-dstAccessMask-parameter
                // VUID-VkBufferMemoryBarrier2-srcAccessMask-parameter
                // VUID-VkBufferMemoryBarrier2-dstAccessMask-parameter
                // VUID-VkImageMemoryBarrier2-srcAccessMask-parameter
                // VUID-VkImageMemoryBarrier2-dstAccessMask-parameter
                access.validate_device(device)?;

                // VUID-vkCmdPipelineBarrier2-srcStageMask-03849
                // VUID-vkCmdPipelineBarrier2-dstStageMask-03850
                if !PipelineStages::from(queue_family_properties.queue_flags).contains(stages) {
                    match ty {
                        'm' => {
                            return Err(SynchronizationError::MemoryBarrierStageNotSupported {
                                barrier_index,
                            })
                        }
                        'b' => {
                            return Err(
                                SynchronizationError::BufferMemoryBarrierStageNotSupported {
                                    barrier_index,
                                },
                            )
                        }
                        'i' => {
                            return Err(SynchronizationError::ImageMemoryBarrierStageNotSupported {
                                barrier_index,
                            })
                        }
                        _ => unreachable!(),
                    }
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03929
                // VUID-VkMemoryBarrier2-dstStageMask-03929
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03929
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03929
                // VUID-VkImageMemoryBarrier2-srcStageMask-03930
                // VUID-VkImageMemoryBarrier2-dstStageMask-03930
                if stages.intersects(PipelineStages::GEOMETRY_SHADER)
                    && !device.enabled_features().geometry_shader
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::GEOMETRY_SHADER`",
                        requires_one_of: RequiresOneOf {
                            features: &["geometry_shader"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03930
                // VUID-VkMemoryBarrier2-dstStageMask-03930
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03930
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03930
                // VUID-VkImageMemoryBarrier2-srcStageMask-03930
                // VUID-VkImageMemoryBarrier2-dstStageMask-03930
                if stages.intersects(
                    PipelineStages::TESSELLATION_CONTROL_SHADER
                        | PipelineStages::TESSELLATION_EVALUATION_SHADER,
                ) && !device.enabled_features().tessellation_shader
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                            `PipelineStages::TESSELLATION_EVALUATION_SHADER`",
                        requires_one_of: RequiresOneOf {
                            features: &["tessellation_shader"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03931
                // VUID-VkMemoryBarrier2-dstStageMask-03931
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03931
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03931
                // VUID-VImagekMemoryBarrier2-srcStageMask-03931
                // VUID-VkImageMemoryBarrier2-dstStageMask-03931
                if stages.intersects(PipelineStages::CONDITIONAL_RENDERING)
                    && !device.enabled_features().conditional_rendering
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::CONDITIONAL_RENDERING`",
                        requires_one_of: RequiresOneOf {
                            features: &["conditional_rendering"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03932
                // VUID-VkMemoryBarrier2-dstStageMask-03932
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03932
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03932
                // VUID-VkImageMemoryBarrier2-srcStageMask-03932
                // VUID-VkImageMemoryBarrier2-dstStageMask-03932
                if stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS)
                    && !device.enabled_features().fragment_density_map
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`",
                        requires_one_of: RequiresOneOf {
                            features: &["fragment_density_map"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03933
                // VUID-VkMemoryBarrier2-dstStageMask-03933
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03933
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03933
                // VUID-VkImageMemoryBarrier2-srcStageMask-03933
                // VUID-VkImageMemoryBarrier2-dstStageMask-03933
                if stages.intersects(PipelineStages::TRANSFORM_FEEDBACK)
                    && !device.enabled_features().transform_feedback
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::TRANSFORM_FEEDBACK`",
                        requires_one_of: RequiresOneOf {
                            features: &["transform_feedback"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03934
                // VUID-VkMemoryBarrier2-dstStageMask-03934
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03934
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03934
                // VUID-VkImageMemoryBarrier2-srcStageMask-03934
                // VUID-VkImageMemoryBarrier2-dstStageMask-03934
                if stages.intersects(PipelineStages::MESH_SHADER)
                    && !device.enabled_features().mesh_shader
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::MESH_SHADER`",
                        requires_one_of: RequiresOneOf {
                            features: &["mesh_shader"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03935
                // VUID-VkMemoryBarrier2-dstStageMask-03935
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03935
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03935
                // VUID-VkImageMemoryBarrier2-srcStageMask-03935
                // VUID-VkImageMemoryBarrier2-dstStageMask-03935
                if stages.intersects(PipelineStages::TASK_SHADER)
                    && !device.enabled_features().task_shader
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::TASK_SHADER`",
                        requires_one_of: RequiresOneOf {
                            features: &["task_shader"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-shadingRateImage-07316
                // VUID-VkMemoryBarrier2-shadingRateImage-07316
                // VUID-VkBufferMemoryBarrier2-shadingRateImage-07316
                // VUID-VkBufferMemoryBarrier2-shadingRateImage-07316
                // VUID-VkImageMemoryBarrier2-shadingRateImage-07316
                // VUID-VkImageMemoryBarrier2-shadingRateImage-07316
                if stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT)
                    && !(device.enabled_features().attachment_fragment_shading_rate
                        || device.enabled_features().shading_rate_image)
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`",
                        requires_one_of: RequiresOneOf {
                            features: &["attachment_fragment_shading_rate", "shading_rate_image"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-04957
                // VUID-VkMemoryBarrier2-dstStageMask-04957
                // VUID-VkBufferMemoryBarrier2-srcStageMask-04957
                // VUID-VkBufferMemoryBarrier2-dstStageMask-04957
                // VUID-VkImageMemoryBarrier2-srcStageMask-04957
                // VUID-VkImageMemoryBarrier2-dstStageMask-04957
                if stages.intersects(PipelineStages::SUBPASS_SHADING)
                    && !device.enabled_features().subpass_shading
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::SUBPASS_SHADING`",
                        requires_one_of: RequiresOneOf {
                            features: &["subpass_shading"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-04995
                // VUID-VkMemoryBarrier2-dstStageMask-04995
                // VUID-VkBufferMemoryBarrier2-srcStageMask-04995
                // VUID-VkBufferMemoryBarrier2-dstStageMask-04995
                // VUID-VkImageMemoryBarrier2-srcStageMask-04995
                // VUID-VkImageMemoryBarrier2-dstStageMask-04995
                if stages.intersects(PipelineStages::INVOCATION_MASK)
                    && !device.enabled_features().invocation_mask
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::INVOCATION_MASK`",
                        requires_one_of: RequiresOneOf {
                            features: &["invocation_mask"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-vkCmdPipelineBarrier-srcStageMask-03937
                // VUID-vkCmdPipelineBarrier-dstStageMask-03937
                if stages.is_empty() && !device.enabled_features().synchronization2 {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            is empty",
                        requires_one_of: RequiresOneOf {
                            features: &["synchronization2"],
                            ..Default::default()
                        },
                    });
                }

                // A bit of a ridiculous number of VUIDs...

                // VUID-VkMemoryBarrier2-srcAccessMask-03900
                // ..
                // VUID-VkMemoryBarrier2-srcAccessMask-07458

                // VUID-VkMemoryBarrier2-dstAccessMask-03900
                // ..
                // VUID-VkMemoryBarrier2-dstAccessMask-07458

                // VUID-VkBufferMemoryBarrier2-srcAccessMask-03900
                // ..
                // VUID-VkBufferMemoryBarrier2-srcAccessMask-07458

                // VUID-VkBufferMemoryBarrier2-dstAccessMask-03900
                // ..
                // VUID-VkBufferMemoryBarrier2-dstAccessMask-07458

                // VUID-VkImageMemoryBarrier2-srcAccessMask-03900
                // ..
                // VUID-VkImageMemoryBarrier2-srcAccessMask-07458

                // VUID-VkImageMemoryBarrier2-dstAccessMask-03900
                // ..
                // VUID-VkImageMemoryBarrier2-dstAccessMask-07458

                if !AccessFlags::from(stages).contains(access) {
                    match ty {
                        'm' => {
                            return Err(
                                SynchronizationError::MemoryBarrierAccessNotSupportedByStages {
                                    barrier_index,
                                },
                            )
                        }
                        'b' => return Err(
                            SynchronizationError::BufferMemoryBarrierAccessNotSupportedByStages {
                                barrier_index,
                            },
                        ),
                        'i' => return Err(
                            SynchronizationError::ImageMemoryBarrierAccessNotSupportedByStages {
                                barrier_index,
                            },
                        ),
                        _ => unreachable!(),
                    }
                }
            }

            // VUID-VkMemoryBarrier2-srcAccessMask-06256
            // VUID-VkBufferMemoryBarrier2-srcAccessMask-06256
            // VUID-VkImageMemoryBarrier2-srcAccessMask-06256
            if !device.enabled_features().ray_query
                && src_access.intersects(AccessFlags::ACCELERATION_STRUCTURE_READ)
                && src_stages.intersects(
                    PipelineStages::VERTEX_SHADER
                        | PipelineStages::TESSELLATION_CONTROL_SHADER
                        | PipelineStages::TESSELLATION_EVALUATION_SHADER
                        | PipelineStages::GEOMETRY_SHADER
                        | PipelineStages::FRAGMENT_SHADER
                        | PipelineStages::COMPUTE_SHADER
                        | PipelineStages::PRE_RASTERIZATION_SHADERS
                        | PipelineStages::TASK_SHADER
                        | PipelineStages::MESH_SHADER,
                )
            {
                return Err(SynchronizationError::RequirementNotMet {
                    required_for: "One of `dependency_info.memory_barriers`, \
                        `dependency_info.buffer_memory_barriers` or \
                        `dependency_info.image_memory_barriers` has an element where \
                        `src_access` contains `ACCELERATION_STRUCTURE_READ`, and \
                        `src_stages` contains a shader stage other than `RAY_TRACING_SHADER`",
                    requires_one_of: RequiresOneOf {
                        features: &["ray_query"],
                        ..Default::default()
                    },
                });
            }

            Ok(())
        };

        let check_queue_family_ownership_transfer = |ty: char,
                                                     barrier_index: usize,
                                                     src_stages: PipelineStages,
                                                     dst_stages: PipelineStages,
                                                     queue_family_ownership_transfer: Option<
            QueueFamilyOwnershipTransfer,
        >,
                                                     sharing: &Sharing<_>|
         -> Result<(), SynchronizationError> {
            if let Some(transfer) = queue_family_ownership_transfer {
                // VUID?
                transfer.validate_device(device)?;

                // VUID-VkBufferMemoryBarrier2-srcQueueFamilyIndex-04087
                // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04070
                // Ensured by the definition of `QueueFamilyOwnershipTransfer`.

                // VUID-VkBufferMemoryBarrier2-buffer-04088
                // VUID-VkImageMemoryBarrier2-image-04071
                // Ensured by the definition of `QueueFamilyOwnershipTransfer`.

                let queue_family_count =
                    device.physical_device().queue_family_properties().len() as u32;

                let provided_queue_family_index = match (sharing, transfer) {
                    (
                        Sharing::Exclusive,
                        QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                            src_index,
                            dst_index,
                        },
                    ) => Some(max(src_index, dst_index)),
                    (
                        Sharing::Exclusive,
                        QueueFamilyOwnershipTransfer::ExclusiveToExternal { src_index }
                        | QueueFamilyOwnershipTransfer::ExclusiveToForeign { src_index },
                    ) => Some(src_index),
                    (
                        Sharing::Exclusive,
                        QueueFamilyOwnershipTransfer::ExclusiveFromExternal { dst_index }
                        | QueueFamilyOwnershipTransfer::ExclusiveFromForeign { dst_index },
                    ) => Some(dst_index),
                    (
                        Sharing::Concurrent(_),
                        QueueFamilyOwnershipTransfer::ConcurrentToExternal
                        | QueueFamilyOwnershipTransfer::ConcurrentFromExternal
                        | QueueFamilyOwnershipTransfer::ConcurrentToForeign
                        | QueueFamilyOwnershipTransfer::ConcurrentFromForeign,
                    ) => None,
                    _ => match ty {
                        'b' => return Err(SynchronizationError::BufferMemoryBarrierOwnershipTransferSharingMismatch {
                            barrier_index,
                        }),
                        'i' => return Err(SynchronizationError::ImageMemoryBarrierOwnershipTransferSharingMismatch {
                            barrier_index,
                        }),
                        _ => unreachable!(),
                    },
                }.filter(|&index| index >= queue_family_count);

                // VUID-VkBufferMemoryBarrier2-buffer-04089
                // VUID-VkImageMemoryBarrier2-image-04072

                if let Some(provided_queue_family_index) = provided_queue_family_index {
                    match ty {
                        'b' => return Err(SynchronizationError::BufferMemoryBarrierOwnershipTransferIndexOutOfRange {
                            barrier_index,
                            provided_queue_family_index,
                            queue_family_count,
                        }),
                        'i' => return Err(SynchronizationError::ImageMemoryBarrierOwnershipTransferIndexOutOfRange {
                            barrier_index,
                            provided_queue_family_index,
                            queue_family_count,
                        }),
                        _ => unreachable!(),
                    }
                }

                // VUID-VkBufferMemoryBarrier2-srcStageMask-03851
                // VUID-VkImageMemoryBarrier2-srcStageMask-03854
                if src_stages.intersects(PipelineStages::HOST)
                    || dst_stages.intersects(PipelineStages::HOST)
                {
                    match ty {
                        'b' => return Err(SynchronizationError::BufferMemoryBarrierOwnershipTransferHostNotAllowed {
                            barrier_index,
                        }),
                        'i' => return Err(SynchronizationError::ImageMemoryBarrierOwnershipTransferHostForbidden {
                            barrier_index,
                        }),
                        _ => unreachable!(),
                    }
                }
            }

            Ok(())
        };

        for (barrier_index, barrier) in memory_barriers.iter().enumerate() {
            let &MemoryBarrier {
                src_stages,
                src_access,
                dst_stages,
                dst_access,
                _ne: _,
            } = barrier;

            /*
                Check stages and access
            */

            check_stages_access(
                'm',
                barrier_index,
                src_stages,
                src_access,
                dst_stages,
                dst_access,
            )?;
        }

        for (barrier_index, barrier) in buffer_memory_barriers.iter().enumerate() {
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

            // VUID-VkBufferMemoryBarrier2-buffer-01931
            // Ensured by Buffer type construction.

            /*
                Check stages and access
            */

            check_stages_access(
                'b',
                barrier_index,
                src_stages,
                src_access,
                dst_stages,
                dst_access,
            )?;

            /*
                Check queue family transfer
            */

            check_queue_family_ownership_transfer(
                'b',
                barrier_index,
                src_stages,
                dst_stages,
                queue_family_ownership_transfer,
                buffer.sharing(),
            )?;

            /*
                Check range
            */

            // VUID-VkBufferMemoryBarrier2-size-01188
            assert!(!range.is_empty());

            // VUID-VkBufferMemoryBarrier2-offset-01187
            // VUID-VkBufferMemoryBarrier2-size-01189
            if range.end > buffer.size() {
                return Err(SynchronizationError::BufferMemoryBarrierOutOfRange {
                    barrier_index,
                    range_end: range.end,
                    buffer_size: buffer.size(),
                });
            }
        }

        for (barrier_index, barrier) in image_memory_barriers.iter().enumerate() {
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

            // VUID-VkImageMemoryBarrier2-image-01932
            // Ensured by Image type construction.

            /*
                Check stages and access
            */

            check_stages_access(
                'i',
                barrier_index,
                src_stages,
                src_access,
                dst_stages,
                dst_access,
            )?;

            /*
                Check layouts
            */

            // VUID-VkImageMemoryBarrier2-oldLayout-parameter
            old_layout.validate_device(device)?;

            // VUID-VkImageMemoryBarrier2-newLayout-parameter
            new_layout.validate_device(device)?;

            // VUID-VkImageMemoryBarrier2-srcStageMask-03855
            if src_stages.intersects(PipelineStages::HOST)
                && !matches!(
                    old_layout,
                    ImageLayout::Preinitialized | ImageLayout::Undefined | ImageLayout::General
                )
            {
                return Err(
                    SynchronizationError::ImageMemoryBarrierOldLayoutFromHostInvalid {
                        barrier_index,
                        old_layout,
                    },
                );
            }

            // VUID-VkImageMemoryBarrier2-oldLayout-01197
            // Not checked yet, therefore unsafe.

            // VUID-VkImageMemoryBarrier2-newLayout-01198
            if matches!(
                new_layout,
                ImageLayout::Undefined | ImageLayout::Preinitialized
            ) {
                return Err(SynchronizationError::ImageMemoryBarrierNewLayoutInvalid {
                    barrier_index,
                });
            }

            // VUID-VkImageMemoryBarrier2-attachmentFeedbackLoopLayout-07313
            /*if !device.enabled_features().attachment_feedback_loop_layout
                && matches!(new_layout, ImageLayout::AttachmentFeedbackLoopOptimal)
            {
                return Err(SynchronizationError::RequirementNotMet {
                    required_for: "`dependency_info.image_memory_barriers` has an element where \
                        `new_layout` is `AttachmentFeedbackLoopOptimal`",
                    requires_one_of: RequiresOneOf {
                        features: &["attachment_feedback_loop_layout"],
                        ..Default::default()
                    },
                });
            }*/

            for layout in [old_layout, new_layout] {
                // VUID-VkImageMemoryBarrier2-synchronization2-06911
                /*if !device.enabled_features().synchronization2
                    && matches!(
                        layout,
                        ImageLayout::AttachmentOptimal | ImageLayout::ReadOnlyOptimal
                    )
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "`dependency_info.image_memory_barriers` has an element \
                            where `old_layout` or `new_layout` is `AttachmentOptimal` or \
                            `ReadOnlyOptimal`",
                        requires_one_of: RequiresOneOf {
                            features: &["synchronization2"],
                            ..Default::default()
                        },
                    });
                }*/

                // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-07006
                /*if layout == ImageLayout::AttachmentFeedbackLoopOptimal {
                    if !image.usage().intersects(
                        ImageUsage::COLOR_ATTACHMENT | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                    ) {
                        return Err(
                            SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                                barrier_index,
                                layout,
                                requires_one_of_usage: ImageUsage::COLOR_ATTACHMENT
                                    | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                            },
                        );
                    }

                    if !image
                        .usage()
                        .intersects(ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED)
                    {
                        return Err(
                            SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                                barrier_index,
                                layout,
                                requires_one_of_usage: ImageUsage::INPUT_ATTACHMENT
                                    | ImageUsage::SAMPLED,
                            },
                        );
                    }

                    if !image
                        .usage()
                        .intersects(ImageUsage::ATTACHMENT_FEEDBACK_LOOP)
                    {
                        return Err(
                            SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                                barrier_index,
                                layout,
                                requires_one_of_usage: ImageUsage::ATTACHMENT_FEEDBACK_LOOP,
                            },
                        );
                    }
                }*/

                let requires_one_of_usage = match layout {
                    // VUID-VkImageMemoryBarrier2-oldLayout-01208
                    ImageLayout::ColorAttachmentOptimal => ImageUsage::COLOR_ATTACHMENT,

                    // VUID-VkImageMemoryBarrier2-oldLayout-01209
                    ImageLayout::DepthStencilAttachmentOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-oldLayout-01210
                    ImageLayout::DepthStencilReadOnlyOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-oldLayout-01211
                    ImageLayout::ShaderReadOnlyOptimal => {
                        ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-oldLayout-01212
                    ImageLayout::TransferSrcOptimal => ImageUsage::TRANSFER_SRC,

                    // VUID-VkImageMemoryBarrier2-oldLayout-01213
                    ImageLayout::TransferDstOptimal => ImageUsage::TRANSFER_DST,

                    // VUID-VkImageMemoryBarrier2-oldLayout-01658
                    ImageLayout::DepthReadOnlyStencilAttachmentOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-oldLayout-01659
                    ImageLayout::DepthAttachmentStencilReadOnlyOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    }

                    /*
                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04065
                    ImageLayout::DepthReadOnlyOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::SAMPLED
                            | ImageUsage::INPUT_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04066
                    ImageLayout::DepthAttachmentOptimal => ImageUsage::DEPTH_STENCIL_ATTACHMENT,

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04067
                    ImageLayout::StencilReadOnlyOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::SAMPLED
                            | ImageUsage::INPUT_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04068
                    ImageLayout::StencilAttachmentOptimal => ImageUsage::DEPTH_STENCIL_ATTACHMENT,

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-03938
                    ImageLayout::AttachmentOptimal => {
                        ImageUsage::COLOR_ATTACHMENT | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-03939
                    ImageLayout::ReadOnlyOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::SAMPLED
                            | ImageUsage::INPUT_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-oldLayout-02088
                    ImageLayout::FragmentShadingRateAttachmentOptimal => {
                        ImageUsage::FRAGMENT_SHADING_RATE_ATTACHMENT
                    }
                     */
                    _ => continue,
                };

                if !image.usage().intersects(requires_one_of_usage) {
                    return Err(
                        SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                            barrier_index,
                            layout,
                            requires_one_of_usage,
                        },
                    );
                }
            }

            /*
                Check queue family tansfer
            */

            check_queue_family_ownership_transfer(
                'i',
                barrier_index,
                src_stages,
                dst_stages,
                queue_family_ownership_transfer,
                image.sharing(),
            )?;

            /*
                Check subresource range
            */

            // VUID-VkImageSubresourceRange-aspectMask-requiredbitmask
            assert!(!subresource_range.aspects.is_empty());

            // VUID-VkImageSubresourceRange-aspectMask-parameter
            subresource_range.aspects.validate_device(device)?;

            let image_aspects = image.format().unwrap().aspects();

            // VUID-VkImageMemoryBarrier2-image-01673
            // VUID-VkImageMemoryBarrier2-image-03319
            if image_aspects.contains(subresource_range.aspects) {
                return Err(SynchronizationError::ImageMemoryBarrierAspectsNotAllowed {
                    barrier_index,
                    aspects: subresource_range.aspects - image_aspects,
                });
            }

            if image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
                // VUID-VkImageMemoryBarrier2-image-03320
                if !device.enabled_features().separate_depth_stencil_layouts
                    && image_aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                    && !subresource_range
                        .aspects
                        .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "`dependency_info.image_memory_barriers` has an element \
                            where `image` has both a depth and a stencil aspect, and \
                            `subresource_range.aspects` does not contain both aspects",
                        requires_one_of: RequiresOneOf {
                            features: &["separate_depth_stencil_layouts"],
                            ..Default::default()
                        },
                    });
                }
            } else {
                // VUID-VkImageMemoryBarrier2-image-01671
                if !image.flags().intersects(ImageCreateFlags::DISJOINT)
                    && subresource_range.aspects != ImageAspects::COLOR
                {
                    return Err(SynchronizationError::ImageMemoryBarrierAspectsNotAllowed {
                        barrier_index,
                        aspects: subresource_range.aspects - ImageAspects::COLOR,
                    });
                }
            }

            // VUID-VkImageSubresourceRange-levelCount-01720
            assert!(!subresource_range.mip_levels.is_empty());

            // VUID-VkImageMemoryBarrier2-subresourceRange-01486
            // VUID-VkImageMemoryBarrier2-subresourceRange-01724
            if subresource_range.mip_levels.end > image.mip_levels() {
                return Err(
                    SynchronizationError::ImageMemoryBarrierMipLevelsOutOfRange {
                        barrier_index,
                        mip_levels_range_end: subresource_range.mip_levels.end,
                        image_mip_levels: image.mip_levels(),
                    },
                );
            }

            // VUID-VkImageSubresourceRange-layerCount-01721
            assert!(!subresource_range.array_layers.is_empty());

            // VUID-VkImageMemoryBarrier2-subresourceRange-01488
            // VUID-VkImageMemoryBarrier2-subresourceRange-01725
            if subresource_range.array_layers.end > image.dimensions().array_layers() {
                return Err(
                    SynchronizationError::ImageMemoryBarrierArrayLayersOutOfRange {
                        barrier_index,
                        array_layers_range_end: subresource_range.array_layers.end,
                        image_array_layers: image.dimensions().array_layers(),
                    },
                );
            }
        }

        /*
            Checks for current render pass
        */

        if let Some(render_pass_state) = self.builder_state.render_pass.as_ref() {
            // VUID-vkCmdPipelineBarrier2-None-06191
            let begin_render_pass_state = match &render_pass_state.render_pass {
                RenderPassStateType::BeginRenderPass(x) => x,
                RenderPassStateType::BeginRendering(_) => {
                    return Err(SynchronizationError::ForbiddenWithBeginRendering)
                }
            };
            let subpass_index = begin_render_pass_state.subpass.index();
            let subpass_desc = begin_render_pass_state.subpass.subpass_desc();
            let dependencies = begin_render_pass_state.subpass.render_pass().dependencies();

            // VUID-vkCmdPipelineBarrier2-pDependencies-02285
            // TODO: see https://github.com/KhronosGroup/Vulkan-Docs/issues/1982
            if !dependencies.iter().any(|dependency| {
                dependency.src_subpass == Some(subpass_index)
                    && dependency.dst_subpass == Some(subpass_index)
            }) {
                return Err(SynchronizationError::MemoryBarrierNoMatchingSubpassSelfDependency);
            }

            // VUID-vkCmdPipelineBarrier2-bufferMemoryBarrierCount-01178
            if !buffer_memory_barriers.is_empty() {
                return Err(SynchronizationError::BufferMemoryBarrierForbiddenInsideRenderPass);
            }

            for (barrier_index, barrier) in image_memory_barriers.iter().enumerate() {
                // VUID-vkCmdPipelineBarrier2-image-04073
                // TODO: How are you supposed to verify this in secondary command buffers,
                // when there is no inherited framebuffer?
                // The image is not known until you execute it in a primary command buffer.
                if let Some(framebuffer) = &begin_render_pass_state.framebuffer {
                    let attachment_index = (framebuffer.attachments().iter())
                        .position(|attachment| attachment.image().inner() == &barrier.image)
                        .ok_or(SynchronizationError::ImageMemoryBarrierNotInputAttachment {
                            barrier_index,
                        })? as u32;

                    if !(subpass_desc.input_attachments.iter().flatten())
                        .any(|atch_ref| atch_ref.attachment == attachment_index)
                    {
                        return Err(SynchronizationError::ImageMemoryBarrierNotInputAttachment {
                            barrier_index,
                        });
                    }

                    if !(subpass_desc.color_attachments.iter().flatten())
                        .chain(subpass_desc.depth_stencil_attachment.as_ref())
                        .any(|atch_ref| atch_ref.attachment == attachment_index)
                    {
                        return Err(SynchronizationError::ImageMemoryBarrierNotColorDepthStencilAttachment {
                            barrier_index,
                        });
                    }
                }

                // VUID-vkCmdPipelineBarrier2-oldLayout-01181
                if barrier.old_layout != barrier.new_layout {
                    return Err(SynchronizationError::ImageMemoryBarrierLayoutTransitionForbiddenInsideRenderPass {
                        barrier_index,
                    });
                }

                // VUID-vkCmdPipelineBarrier2-srcQueueFamilyIndex-01182
                if barrier.queue_family_ownership_transfer.is_some() {
                    return Err(SynchronizationError::ImageMemoryBarrierOwnershipTransferForbiddenInsideRenderPass {
                        barrier_index,
                    });
                }
            }
        } else {
            // VUID-vkCmdPipelineBarrier2-dependencyFlags-01186
            if dependency_flags.intersects(DependencyFlags::VIEW_LOCAL) {
                return Err(SynchronizationError::DependencyFlagsViewLocalNotAllowed);
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn pipeline_barrier_unchecked(
        &mut self,
        dependency_info: DependencyInfo,
    ) -> &mut Self {
        if dependency_info.is_empty() {
            return self;
        }

        let &DependencyInfo {
            dependency_flags,
            ref memory_barriers,
            ref buffer_memory_barriers,
            ref image_memory_barriers,
            _ne: _,
        } = &dependency_info;

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
                debug_assert!(self.device().enabled_extensions().khr_synchronization2);
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

        let command_index = self.next_command_index;
        let command_name = "pipeline_barrier";
        self.resources_usage_state.record_pipeline_barrier(
            command_index,
            command_name,
            &dependency_info,
            self.queue_family_properties().queue_flags,
        );

        self.resources
            .reserve(buffer_memory_barriers.len() + image_memory_barriers.len());

        for barrier in dependency_info.buffer_memory_barriers {
            self.resources.push(Box::new(barrier.buffer));
        }

        for barrier in dependency_info.image_memory_barriers {
            self.resources.push(Box::new(barrier.image));
        }

        self.next_command_index += 1;
        self
    }

    /// Signals an [`Event`] from the device.
    ///
    /// # Safety
    ///
    /// - All images that are accessed by the command must be in the expected image layout.
    /// - For each element of `dependency_info.image_memory_barriers` that contains an image layout
    ///   transition, which is a write operation, the barrier must be defined appropriately to
    ///   ensure no memory access hazard occurs.
    #[inline]
    pub unsafe fn set_event(
        &mut self,
        event: Arc<Event>,
        dependency_info: DependencyInfo,
    ) -> Result<&mut Self, SynchronizationError> {
        self.validate_set_event(&event, &dependency_info)?;

        unsafe { Ok(self.set_event_unchecked(event, dependency_info)) }
    }

    fn validate_set_event(
        &self,
        event: &Event,
        dependency_info: &DependencyInfo,
    ) -> Result<(), SynchronizationError> {
        // VUID-vkCmdSetEvent2-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(SynchronizationError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdSetEvent2-commandBuffer-03826
        // TODO:

        let device = self.device();
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetEvent2-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(SynchronizationError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetEvent2-commonparent
        assert_eq!(device, event.device());

        let &DependencyInfo {
            dependency_flags,
            ref memory_barriers,
            ref buffer_memory_barriers,
            ref image_memory_barriers,
            _ne: _,
        } = dependency_info;

        // VUID-VkDependencyInfo-dependencyFlags-parameter
        dependency_flags.validate_device(device)?;

        // VUID-vkCmdSetEvent2-dependencyFlags-03825
        assert!(dependency_flags.is_empty());

        let check_stages_access = |ty: char,
                                   barrier_index: usize,
                                   src_stages: PipelineStages,
                                   src_access: AccessFlags,
                                   dst_stages: PipelineStages,
                                   dst_access: AccessFlags|
         -> Result<(), SynchronizationError> {
            for (stages, access) in [(src_stages, src_access), (dst_stages, dst_access)] {
                // VUID-vkCmdSetEvent2-synchronization2-03824
                if !device.enabled_features().synchronization2 {
                    if stages.is_2() {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                                `dependency_info.buffer_memory_barriers` or \
                                `dependency_info.image_memory_barriers` has an element where \
                                `src_stages` or `dst_stages` contains flags from \
                                `VkPipelineStageFlagBits2`",
                            requires_one_of: RequiresOneOf {
                                features: &["synchronization2"],
                                ..Default::default()
                            },
                        });
                    }

                    if access.is_2() {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                                `dependency_info.buffer_memory_barriers` or \
                                `dependency_info.image_memory_barriers` has an element where \
                                `src_access` or `dst_access` contains flags from \
                                `VkAccessFlagBits2`",
                            requires_one_of: RequiresOneOf {
                                features: &["synchronization2"],
                                ..Default::default()
                            },
                        });
                    }
                }

                // VUID-VkMemoryBarrier2-srcStageMask-parameter
                // VUID-VkMemoryBarrier2-dstStageMask-parameter
                // VUID-VkBufferMemoryBarrier2-srcStageMask-parameter
                // VUID-VkBufferMemoryBarrier2-dstStageMask-parameter
                // VUID-VkImageMemoryBarrier2-srcStageMask-parameter
                // VUID-VkImageMemoryBarrier2-dstStageMask-parameter
                stages.validate_device(device)?;

                // VUID-VkMemoryBarrier2-srcAccessMask-parameter
                // VUID-VkMemoryBarrier2-dstAccessMask-parameter
                // VUID-VkBufferMemoryBarrier2-srcAccessMask-parameter
                // VUID-VkBufferMemoryBarrier2-dstAccessMask-parameter
                // VUID-VkImageMemoryBarrier2-srcAccessMask-parameter
                // VUID-VkImageMemoryBarrier2-dstAccessMask-parameter
                access.validate_device(device)?;

                // VUID-vkCmdSetEvent2-srcStageMask-03827
                // VUID-vkCmdSetEvent2-dstStageMask-03828
                if !PipelineStages::from(queue_family_properties.queue_flags).contains(stages) {
                    match ty {
                        'm' => {
                            return Err(SynchronizationError::MemoryBarrierStageNotSupported {
                                barrier_index,
                            })
                        }
                        'b' => {
                            return Err(
                                SynchronizationError::BufferMemoryBarrierStageNotSupported {
                                    barrier_index,
                                },
                            )
                        }
                        'i' => {
                            return Err(SynchronizationError::ImageMemoryBarrierStageNotSupported {
                                barrier_index,
                            })
                        }
                        _ => unreachable!(),
                    }
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03929
                // VUID-VkMemoryBarrier2-dstStageMask-03929
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03929
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03929
                // VUID-VkImageMemoryBarrier2-srcStageMask-03930
                // VUID-VkImageMemoryBarrier2-dstStageMask-03930
                if stages.intersects(PipelineStages::GEOMETRY_SHADER)
                    && !device.enabled_features().geometry_shader
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::GEOMETRY_SHADER`",
                        requires_one_of: RequiresOneOf {
                            features: &["geometry_shader"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03930
                // VUID-VkMemoryBarrier2-dstStageMask-03930
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03930
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03930
                // VUID-VkImageMemoryBarrier2-srcStageMask-03930
                // VUID-VkImageMemoryBarrier2-dstStageMask-03930
                if stages.intersects(
                    PipelineStages::TESSELLATION_CONTROL_SHADER
                        | PipelineStages::TESSELLATION_EVALUATION_SHADER,
                ) && !device.enabled_features().tessellation_shader
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                            `PipelineStages::TESSELLATION_EVALUATION_SHADER`",
                        requires_one_of: RequiresOneOf {
                            features: &["tessellation_shader"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03931
                // VUID-VkMemoryBarrier2-dstStageMask-03931
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03931
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03931
                // VUID-VImagekMemoryBarrier2-srcStageMask-03931
                // VUID-VkImageMemoryBarrier2-dstStageMask-03931
                if stages.intersects(PipelineStages::CONDITIONAL_RENDERING)
                    && !device.enabled_features().conditional_rendering
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::CONDITIONAL_RENDERING`",
                        requires_one_of: RequiresOneOf {
                            features: &["conditional_rendering"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03932
                // VUID-VkMemoryBarrier2-dstStageMask-03932
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03932
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03932
                // VUID-VkImageMemoryBarrier2-srcStageMask-03932
                // VUID-VkImageMemoryBarrier2-dstStageMask-03932
                if stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS)
                    && !device.enabled_features().fragment_density_map
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`",
                        requires_one_of: RequiresOneOf {
                            features: &["fragment_density_map"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03933
                // VUID-VkMemoryBarrier2-dstStageMask-03933
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03933
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03933
                // VUID-VkImageMemoryBarrier2-srcStageMask-03933
                // VUID-VkImageMemoryBarrier2-dstStageMask-03933
                if stages.intersects(PipelineStages::TRANSFORM_FEEDBACK)
                    && !device.enabled_features().transform_feedback
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::TRANSFORM_FEEDBACK`",
                        requires_one_of: RequiresOneOf {
                            features: &["transform_feedback"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03934
                // VUID-VkMemoryBarrier2-dstStageMask-03934
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03934
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03934
                // VUID-VkImageMemoryBarrier2-srcStageMask-03934
                // VUID-VkImageMemoryBarrier2-dstStageMask-03934
                if stages.intersects(PipelineStages::MESH_SHADER)
                    && !device.enabled_features().mesh_shader
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::MESH_SHADER`",
                        requires_one_of: RequiresOneOf {
                            features: &["mesh_shader"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-03935
                // VUID-VkMemoryBarrier2-dstStageMask-03935
                // VUID-VkBufferMemoryBarrier2-srcStageMask-03935
                // VUID-VkBufferMemoryBarrier2-dstStageMask-03935
                // VUID-VkImageMemoryBarrier2-srcStageMask-03935
                // VUID-VkImageMemoryBarrier2-dstStageMask-03935
                if stages.intersects(PipelineStages::TASK_SHADER)
                    && !device.enabled_features().task_shader
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::TASK_SHADER`",
                        requires_one_of: RequiresOneOf {
                            features: &["task_shader"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-shadingRateImage-07316
                // VUID-VkMemoryBarrier2-shadingRateImage-07316
                // VUID-VkBufferMemoryBarrier2-shadingRateImage-07316
                // VUID-VkBufferMemoryBarrier2-shadingRateImage-07316
                // VUID-VkImageMemoryBarrier2-shadingRateImage-07316
                // VUID-VkImageMemoryBarrier2-shadingRateImage-07316
                if stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT)
                    && !(device.enabled_features().attachment_fragment_shading_rate
                        || device.enabled_features().shading_rate_image)
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`",
                        requires_one_of: RequiresOneOf {
                            features: &["attachment_fragment_shading_rate", "shading_rate_image"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-04957
                // VUID-VkMemoryBarrier2-dstStageMask-04957
                // VUID-VkBufferMemoryBarrier2-srcStageMask-04957
                // VUID-VkBufferMemoryBarrier2-dstStageMask-04957
                // VUID-VkImageMemoryBarrier2-srcStageMask-04957
                // VUID-VkImageMemoryBarrier2-dstStageMask-04957
                if stages.intersects(PipelineStages::SUBPASS_SHADING)
                    && !device.enabled_features().subpass_shading
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::SUBPASS_SHADING`",
                        requires_one_of: RequiresOneOf {
                            features: &["subpass_shading"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkMemoryBarrier2-srcStageMask-04995
                // VUID-VkMemoryBarrier2-dstStageMask-04995
                // VUID-VkBufferMemoryBarrier2-srcStageMask-04995
                // VUID-VkBufferMemoryBarrier2-dstStageMask-04995
                // VUID-VkImageMemoryBarrier2-srcStageMask-04995
                // VUID-VkImageMemoryBarrier2-dstStageMask-04995
                if stages.intersects(PipelineStages::INVOCATION_MASK)
                    && !device.enabled_features().invocation_mask
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::INVOCATION_MASK`",
                        requires_one_of: RequiresOneOf {
                            features: &["invocation_mask"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-vkCmdSetEvent-stageMask-03937
                if stages.is_empty() && !device.enabled_features().synchronization2 {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            is empty",
                        requires_one_of: RequiresOneOf {
                            features: &["synchronization2"],
                            ..Default::default()
                        },
                    });
                }

                // A bit of a ridiculous number of VUIDs...

                // VUID-VkMemoryBarrier2-srcAccessMask-03900
                // ..
                // VUID-VkMemoryBarrier2-srcAccessMask-07458

                // VUID-VkMemoryBarrier2-dstAccessMask-03900
                // ..
                // VUID-VkMemoryBarrier2-dstAccessMask-07458

                // VUID-VkBufferMemoryBarrier2-srcAccessMask-03900
                // ..
                // VUID-VkBufferMemoryBarrier2-srcAccessMask-07458

                // VUID-VkBufferMemoryBarrier2-dstAccessMask-03900
                // ..
                // VUID-VkBufferMemoryBarrier2-dstAccessMask-07458

                // VUID-VkImageMemoryBarrier2-srcAccessMask-03900
                // ..
                // VUID-VkImageMemoryBarrier2-srcAccessMask-07458

                // VUID-VkImageMemoryBarrier2-dstAccessMask-03900
                // ..
                // VUID-VkImageMemoryBarrier2-dstAccessMask-07458

                if !AccessFlags::from(stages).contains(access) {
                    match ty {
                        'm' => {
                            return Err(
                                SynchronizationError::MemoryBarrierAccessNotSupportedByStages {
                                    barrier_index,
                                },
                            )
                        }
                        'b' => return Err(
                            SynchronizationError::BufferMemoryBarrierAccessNotSupportedByStages {
                                barrier_index,
                            },
                        ),
                        'i' => return Err(
                            SynchronizationError::ImageMemoryBarrierAccessNotSupportedByStages {
                                barrier_index,
                            },
                        ),
                        _ => unreachable!(),
                    }
                }
            }

            // VUID-VkMemoryBarrier2-srcAccessMask-06256
            // VUID-VkBufferMemoryBarrier2-srcAccessMask-06256
            // VUID-VkImageMemoryBarrier2-srcAccessMask-06256
            if !device.enabled_features().ray_query
                && src_access.intersects(AccessFlags::ACCELERATION_STRUCTURE_READ)
                && src_stages.intersects(
                    PipelineStages::VERTEX_SHADER
                        | PipelineStages::TESSELLATION_CONTROL_SHADER
                        | PipelineStages::TESSELLATION_EVALUATION_SHADER
                        | PipelineStages::GEOMETRY_SHADER
                        | PipelineStages::FRAGMENT_SHADER
                        | PipelineStages::COMPUTE_SHADER
                        | PipelineStages::PRE_RASTERIZATION_SHADERS
                        | PipelineStages::TASK_SHADER
                        | PipelineStages::MESH_SHADER,
                )
            {
                return Err(SynchronizationError::RequirementNotMet {
                    required_for: "One of `dependency_info.memory_barriers`, \
                        `dependency_info.buffer_memory_barriers` or \
                        `dependency_info.image_memory_barriers` has an element where \
                        `src_access` contains `ACCELERATION_STRUCTURE_READ`, and \
                        `src_stages` contains a shader stage other than `RAY_TRACING_SHADER`",
                    requires_one_of: RequiresOneOf {
                        features: &["ray_query"],
                        ..Default::default()
                    },
                });
            }

            Ok(())
        };

        let check_queue_family_ownership_transfer = |ty: char,
                                                     barrier_index: usize,
                                                     src_stages: PipelineStages,
                                                     dst_stages: PipelineStages,
                                                     queue_family_ownership_transfer: Option<
            QueueFamilyOwnershipTransfer,
        >,
                                                     sharing: &Sharing<_>|
         -> Result<(), SynchronizationError> {
            if let Some(transfer) = queue_family_ownership_transfer {
                // VUID?
                transfer.validate_device(device)?;

                // VUID-VkBufferMemoryBarrier2-srcQueueFamilyIndex-04087
                // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04070
                // Ensured by the definition of `QueueFamilyOwnershipTransfer`.

                // VUID-VkBufferMemoryBarrier2-buffer-04088
                // VUID-VkImageMemoryBarrier2-image-04071
                // Ensured by the definition of `QueueFamilyOwnershipTransfer`.

                let queue_family_count =
                    device.physical_device().queue_family_properties().len() as u32;

                let provided_queue_family_index = match (sharing, transfer) {
                    (
                        Sharing::Exclusive,
                        QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                            src_index,
                            dst_index,
                        },
                    ) => Some(max(src_index, dst_index)),
                    (
                        Sharing::Exclusive,
                        QueueFamilyOwnershipTransfer::ExclusiveToExternal { src_index }
                        | QueueFamilyOwnershipTransfer::ExclusiveToForeign { src_index },
                    ) => Some(src_index),
                    (
                        Sharing::Exclusive,
                        QueueFamilyOwnershipTransfer::ExclusiveFromExternal { dst_index }
                        | QueueFamilyOwnershipTransfer::ExclusiveFromForeign { dst_index },
                    ) => Some(dst_index),
                    (
                        Sharing::Concurrent(_),
                        QueueFamilyOwnershipTransfer::ConcurrentToExternal
                        | QueueFamilyOwnershipTransfer::ConcurrentFromExternal
                        | QueueFamilyOwnershipTransfer::ConcurrentToForeign
                        | QueueFamilyOwnershipTransfer::ConcurrentFromForeign,
                    ) => None,
                    _ => match ty {
                        'b' => return Err(SynchronizationError::BufferMemoryBarrierOwnershipTransferSharingMismatch {
                            barrier_index,
                        }),
                        'i' => return Err(SynchronizationError::ImageMemoryBarrierOwnershipTransferSharingMismatch {
                            barrier_index,
                        }),
                        _ => unreachable!(),
                    },
                }.filter(|&index| index >= queue_family_count);

                // VUID-VkBufferMemoryBarrier2-buffer-04089
                // VUID-VkImageMemoryBarrier2-image-04072

                if let Some(provided_queue_family_index) = provided_queue_family_index {
                    match ty {
                        'b' => return Err(SynchronizationError::BufferMemoryBarrierOwnershipTransferIndexOutOfRange {
                            barrier_index,
                            provided_queue_family_index,
                            queue_family_count,
                        }),
                        'i' => return Err(SynchronizationError::ImageMemoryBarrierOwnershipTransferIndexOutOfRange {
                            barrier_index,
                            provided_queue_family_index,
                            queue_family_count,
                        }),
                        _ => unreachable!(),
                    }
                }

                // VUID-VkBufferMemoryBarrier2-srcStageMask-03851
                // VUID-VkImageMemoryBarrier2-srcStageMask-03854
                if src_stages.intersects(PipelineStages::HOST)
                    || dst_stages.intersects(PipelineStages::HOST)
                {
                    match ty {
                        'b' => return Err(SynchronizationError::BufferMemoryBarrierOwnershipTransferHostNotAllowed {
                            barrier_index,
                        }),
                        'i' => return Err(SynchronizationError::ImageMemoryBarrierOwnershipTransferHostForbidden {
                            barrier_index,
                        }),
                        _ => unreachable!(),
                    }
                }
            }

            Ok(())
        };

        for (barrier_index, barrier) in memory_barriers.iter().enumerate() {
            let &MemoryBarrier {
                src_stages,
                src_access,
                dst_stages,
                dst_access,
                _ne: _,
            } = barrier;

            /*
                Check stages and access
            */

            check_stages_access(
                'm',
                barrier_index,
                src_stages,
                src_access,
                dst_stages,
                dst_access,
            )?;
        }

        for (barrier_index, barrier) in buffer_memory_barriers.iter().enumerate() {
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

            // VUID-VkBufferMemoryBarrier2-buffer-01931
            // Ensured by Buffer type construction.

            /*
                Check stages and access
            */

            check_stages_access(
                'b',
                barrier_index,
                src_stages,
                src_access,
                dst_stages,
                dst_access,
            )?;

            /*
                Check queue family transfer
            */

            check_queue_family_ownership_transfer(
                'b',
                barrier_index,
                src_stages,
                dst_stages,
                queue_family_ownership_transfer,
                buffer.sharing(),
            )?;

            /*
                Check range
            */

            // VUID-VkBufferMemoryBarrier2-size-01188
            assert!(!range.is_empty());

            // VUID-VkBufferMemoryBarrier2-offset-01187
            // VUID-VkBufferMemoryBarrier2-size-01189
            if range.end > buffer.size() {
                return Err(SynchronizationError::BufferMemoryBarrierOutOfRange {
                    barrier_index,
                    range_end: range.end,
                    buffer_size: buffer.size(),
                });
            }
        }

        for (barrier_index, barrier) in image_memory_barriers.iter().enumerate() {
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

            // VUID-VkImageMemoryBarrier2-image-01932
            // Ensured by Image type construction.

            /*
                Check stages and access
            */

            check_stages_access(
                'i',
                barrier_index,
                src_stages,
                src_access,
                dst_stages,
                dst_access,
            )?;

            /*
                Check layouts
            */

            // VUID-VkImageMemoryBarrier2-oldLayout-parameter
            old_layout.validate_device(device)?;

            // VUID-VkImageMemoryBarrier2-newLayout-parameter
            new_layout.validate_device(device)?;

            // VUID-VkImageMemoryBarrier2-srcStageMask-03855
            if src_stages.intersects(PipelineStages::HOST)
                && !matches!(
                    old_layout,
                    ImageLayout::Preinitialized | ImageLayout::Undefined | ImageLayout::General
                )
            {
                return Err(
                    SynchronizationError::ImageMemoryBarrierOldLayoutFromHostInvalid {
                        barrier_index,
                        old_layout,
                    },
                );
            }

            // VUID-VkImageMemoryBarrier2-oldLayout-01197
            // Not checked yet, therefore unsafe.

            // VUID-VkImageMemoryBarrier2-newLayout-01198
            if matches!(
                new_layout,
                ImageLayout::Undefined | ImageLayout::Preinitialized
            ) {
                return Err(SynchronizationError::ImageMemoryBarrierNewLayoutInvalid {
                    barrier_index,
                });
            }

            // VUID-VkImageMemoryBarrier2-attachmentFeedbackLoopLayout-07313
            /*if !device.enabled_features().attachment_feedback_loop_layout
                && matches!(new_layout, ImageLayout::AttachmentFeedbackLoopOptimal)
            {
                return Err(SynchronizationError::RequirementNotMet {
                    required_for: "`dependency_info.image_memory_barriers` has an element where \
                        `new_layout` is `AttachmentFeedbackLoopOptimal`",
                    requires_one_of: RequiresOneOf {
                        features: &["attachment_feedback_loop_layout"],
                        ..Default::default()
                    },
                });
            }*/

            for layout in [old_layout, new_layout] {
                // VUID-VkImageMemoryBarrier2-synchronization2-06911
                /*if !device.enabled_features().synchronization2
                    && matches!(
                        layout,
                        ImageLayout::AttachmentOptimal | ImageLayout::ReadOnlyOptimal
                    )
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "`dependency_info.image_memory_barriers` has an element \
                            where `old_layout` or `new_layout` is `AttachmentOptimal` or \
                            `ReadOnlyOptimal`",
                        requires_one_of: RequiresOneOf {
                            features: &["synchronization2"],
                            ..Default::default()
                        },
                    });
                }*/

                // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-07006
                /*if layout == ImageLayout::AttachmentFeedbackLoopOptimal {
                    if !image.usage().intersects(
                        ImageUsage::COLOR_ATTACHMENT | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                    ) {
                        return Err(
                            SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                                barrier_index,
                                layout,
                                requires_one_of_usage: ImageUsage::COLOR_ATTACHMENT
                                    | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                            },
                        );
                    }

                    if !image
                        .usage()
                        .intersects(ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED)
                    {
                        return Err(
                            SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                                barrier_index,
                                layout,
                                requires_one_of_usage: ImageUsage::INPUT_ATTACHMENT
                                    | ImageUsage::SAMPLED,
                            },
                        );
                    }

                    if !image
                        .usage()
                        .intersects(ImageUsage::ATTACHMENT_FEEDBACK_LOOP)
                    {
                        return Err(
                            SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                                barrier_index,
                                layout,
                                requires_one_of_usage: ImageUsage::ATTACHMENT_FEEDBACK_LOOP,
                            },
                        );
                    }
                }*/

                let requires_one_of_usage = match layout {
                    // VUID-VkImageMemoryBarrier2-oldLayout-01208
                    ImageLayout::ColorAttachmentOptimal => ImageUsage::COLOR_ATTACHMENT,

                    // VUID-VkImageMemoryBarrier2-oldLayout-01209
                    ImageLayout::DepthStencilAttachmentOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-oldLayout-01210
                    ImageLayout::DepthStencilReadOnlyOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-oldLayout-01211
                    ImageLayout::ShaderReadOnlyOptimal => {
                        ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-oldLayout-01212
                    ImageLayout::TransferSrcOptimal => ImageUsage::TRANSFER_SRC,

                    // VUID-VkImageMemoryBarrier2-oldLayout-01213
                    ImageLayout::TransferDstOptimal => ImageUsage::TRANSFER_DST,

                    // VUID-VkImageMemoryBarrier2-oldLayout-01658
                    ImageLayout::DepthReadOnlyStencilAttachmentOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-oldLayout-01659
                    ImageLayout::DepthAttachmentStencilReadOnlyOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    }

                    /*
                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04065
                    ImageLayout::DepthReadOnlyOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::SAMPLED
                            | ImageUsage::INPUT_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04066
                    ImageLayout::DepthAttachmentOptimal => ImageUsage::DEPTH_STENCIL_ATTACHMENT,

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04067
                    ImageLayout::StencilReadOnlyOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::SAMPLED
                            | ImageUsage::INPUT_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04068
                    ImageLayout::StencilAttachmentOptimal => ImageUsage::DEPTH_STENCIL_ATTACHMENT,

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-03938
                    ImageLayout::AttachmentOptimal => {
                        ImageUsage::COLOR_ATTACHMENT | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-03939
                    ImageLayout::ReadOnlyOptimal => {
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::SAMPLED
                            | ImageUsage::INPUT_ATTACHMENT
                    }

                    // VUID-VkImageMemoryBarrier2-oldLayout-02088
                    ImageLayout::FragmentShadingRateAttachmentOptimal => {
                        ImageUsage::FRAGMENT_SHADING_RATE_ATTACHMENT
                    }
                     */
                    _ => continue,
                };

                if !image.usage().intersects(requires_one_of_usage) {
                    return Err(
                        SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                            barrier_index,
                            layout,
                            requires_one_of_usage,
                        },
                    );
                }
            }

            /*
                Check queue family tansfer
            */

            check_queue_family_ownership_transfer(
                'i',
                barrier_index,
                src_stages,
                dst_stages,
                queue_family_ownership_transfer,
                image.sharing(),
            )?;

            /*
                Check subresource range
            */

            // VUID-VkImageSubresourceRange-aspectMask-requiredbitmask
            assert!(!subresource_range.aspects.is_empty());

            // VUID-VkImageSubresourceRange-aspectMask-parameter
            subresource_range.aspects.validate_device(device)?;

            let image_aspects = image.format().unwrap().aspects();

            // VUID-VkImageMemoryBarrier2-image-01673
            // VUID-VkImageMemoryBarrier2-image-03319
            if image_aspects.contains(subresource_range.aspects) {
                return Err(SynchronizationError::ImageMemoryBarrierAspectsNotAllowed {
                    barrier_index,
                    aspects: subresource_range.aspects - image_aspects,
                });
            }

            if image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
                // VUID-VkImageMemoryBarrier2-image-03320
                if !device.enabled_features().separate_depth_stencil_layouts
                    && image_aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                    && !subresource_range
                        .aspects
                        .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "`dependency_info.image_memory_barriers` has an element \
                            where `image` has both a depth and a stencil aspect, and \
                            `subresource_range.aspects` does not contain both aspects",
                        requires_one_of: RequiresOneOf {
                            features: &["separate_depth_stencil_layouts"],
                            ..Default::default()
                        },
                    });
                }
            } else {
                // VUID-VkImageMemoryBarrier2-image-01671
                if !image.flags().intersects(ImageCreateFlags::DISJOINT)
                    && subresource_range.aspects != ImageAspects::COLOR
                {
                    return Err(SynchronizationError::ImageMemoryBarrierAspectsNotAllowed {
                        barrier_index,
                        aspects: subresource_range.aspects - ImageAspects::COLOR,
                    });
                }
            }

            // VUID-VkImageSubresourceRange-levelCount-01720
            assert!(!subresource_range.mip_levels.is_empty());

            // VUID-VkImageMemoryBarrier2-subresourceRange-01486
            // VUID-VkImageMemoryBarrier2-subresourceRange-01724
            if subresource_range.mip_levels.end > image.mip_levels() {
                return Err(
                    SynchronizationError::ImageMemoryBarrierMipLevelsOutOfRange {
                        barrier_index,
                        mip_levels_range_end: subresource_range.mip_levels.end,
                        image_mip_levels: image.mip_levels(),
                    },
                );
            }

            // VUID-VkImageSubresourceRange-layerCount-01721
            assert!(!subresource_range.array_layers.is_empty());

            // VUID-VkImageMemoryBarrier2-subresourceRange-01488
            // VUID-VkImageMemoryBarrier2-subresourceRange-01725
            if subresource_range.array_layers.end > image.dimensions().array_layers() {
                return Err(
                    SynchronizationError::ImageMemoryBarrierArrayLayersOutOfRange {
                        barrier_index,
                        array_layers_range_end: subresource_range.array_layers.end,
                        image_array_layers: image.dimensions().array_layers(),
                    },
                );
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_event_unchecked(
        &mut self,
        event: Arc<Event>,
        dependency_info: DependencyInfo,
    ) -> &mut Self {
        let DependencyInfo {
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
            _ne: _,
        } = dependency_info;

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
                debug_assert!(self.device().enabled_extensions().khr_synchronization2);
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

        self.resources.push(Box::new(event));

        // TODO: sync state update

        self.next_command_index += 1;
        self
    }

    /// Waits for one or more [`Event`]s to be signaled.
    ///
    /// # Safety
    ///
    /// - For each element in `events`, if the event is signaled by [`set_event`], that command
    ///   must have already been recorded or previously submitted to the queue, and the
    ///   `DependencyInfo` provided here must be equal to the `DependencyInfo` used in that
    ///   command.
    /// - For each element in `events`, if the event is signaled by [`Event::set`], that function
    ///   must have already been called before submitting this command to a queue.
    ///
    /// [`set_event`]: Self::set_event
    /// [`Event::set`]: Event::set
    #[inline]
    pub unsafe fn wait_events(
        &mut self,
        events: impl IntoIterator<Item = (Arc<Event>, DependencyInfo)>,
    ) -> Result<&mut Self, SynchronizationError> {
        let events: SmallVec<[(Arc<Event>, DependencyInfo); 4]> = events.into_iter().collect();
        self.validate_wait_events(&events)?;

        unsafe { Ok(self.wait_events_unchecked(events)) }
    }

    fn validate_wait_events(
        &self,
        events: &[(Arc<Event>, DependencyInfo)],
    ) -> Result<(), SynchronizationError> {
        // VUID-vkCmdWaitEvents2-commandBuffer-03846
        // TODO:

        let device = self.device();
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdWaitEvents2-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(SynchronizationError::NotSupportedByQueueFamily);
        }

        if events.is_empty() {
            return Ok(());
        }

        for (event, dependency_info) in events {
            // VUID-vkCmdWaitEvents2-commonparent
            assert_eq!(device, event.device());

            // VUID-vkCmdWaitEvents2-pEvents-03838
            // TODO:

            // VUID-vkCmdWaitEvents2-pEvents-03839
            // TODO:

            // VUID-vkCmdWaitEvents2-pEvents-03840
            // TODO:

            // VUID-vkCmdWaitEvents2-pEvents-03841
            // TODO:

            let &DependencyInfo {
                dependency_flags,
                ref memory_barriers,
                ref buffer_memory_barriers,
                ref image_memory_barriers,
                _ne: _,
            } = dependency_info;

            // VUID-VkDependencyInfo-dependencyFlags-parameter
            dependency_flags.validate_device(device)?;

            let check_stages_access = |ty: char,
                                       barrier_index: usize,
                                       src_stages: PipelineStages,
                                       src_access: AccessFlags,
                                       dst_stages: PipelineStages,
                                       dst_access: AccessFlags|
             -> Result<(), SynchronizationError> {
                for (stages, access) in [(src_stages, src_access), (dst_stages, dst_access)] {
                    // VUID-vkCmdWaitEvents2-synchronization2-03836
                    if !device.enabled_features().synchronization2 {
                        if stages.is_2() {
                            return Err(SynchronizationError::RequirementNotMet {
                                required_for: "One of `dependency_info.memory_barriers`, \
                                `dependency_info.buffer_memory_barriers` or \
                                `dependency_info.image_memory_barriers` has an element where \
                                `src_stages` or `dst_stages` contains flags from \
                                `VkPipelineStageFlagBits2`",
                                requires_one_of: RequiresOneOf {
                                    features: &["synchronization2"],
                                    ..Default::default()
                                },
                            });
                        }

                        if access.is_2() {
                            return Err(SynchronizationError::RequirementNotMet {
                                required_for: "One of `dependency_info.memory_barriers`, \
                                `dependency_info.buffer_memory_barriers` or \
                                `dependency_info.image_memory_barriers` has an element where \
                                `src_access` or `dst_access` contains flags from \
                                `VkAccessFlagBits2`",
                                requires_one_of: RequiresOneOf {
                                    features: &["synchronization2"],
                                    ..Default::default()
                                },
                            });
                        }
                    }

                    // VUID-VkMemoryBarrier2-srcStageMask-parameter
                    // VUID-VkMemoryBarrier2-dstStageMask-parameter
                    // VUID-VkBufferMemoryBarrier2-srcStageMask-parameter
                    // VUID-VkBufferMemoryBarrier2-dstStageMask-parameter
                    // VUID-VkImageMemoryBarrier2-srcStageMask-parameter
                    // VUID-VkImageMemoryBarrier2-dstStageMask-parameter
                    stages.validate_device(device)?;

                    // VUID-VkMemoryBarrier2-srcAccessMask-parameter
                    // VUID-VkMemoryBarrier2-dstAccessMask-parameter
                    // VUID-VkBufferMemoryBarrier2-srcAccessMask-parameter
                    // VUID-VkBufferMemoryBarrier2-dstAccessMask-parameter
                    // VUID-VkImageMemoryBarrier2-srcAccessMask-parameter
                    // VUID-VkImageMemoryBarrier2-dstAccessMask-parameter
                    access.validate_device(device)?;

                    // VUID-vkCmdWaitEvents2-srcStageMask-03842
                    // VUID-vkCmdWaitEvents2-dstStageMask-03843
                    if !PipelineStages::from(queue_family_properties.queue_flags).contains(stages) {
                        match ty {
                            'm' => {
                                return Err(SynchronizationError::MemoryBarrierStageNotSupported {
                                    barrier_index,
                                })
                            }
                            'b' => {
                                return Err(
                                    SynchronizationError::BufferMemoryBarrierStageNotSupported {
                                        barrier_index,
                                    },
                                )
                            }
                            'i' => {
                                return Err(
                                    SynchronizationError::ImageMemoryBarrierStageNotSupported {
                                        barrier_index,
                                    },
                                )
                            }
                            _ => unreachable!(),
                        }
                    }

                    // VUID-VkMemoryBarrier2-srcStageMask-03929
                    // VUID-VkMemoryBarrier2-dstStageMask-03929
                    // VUID-VkBufferMemoryBarrier2-srcStageMask-03929
                    // VUID-VkBufferMemoryBarrier2-dstStageMask-03929
                    // VUID-VkImageMemoryBarrier2-srcStageMask-03930
                    // VUID-VkImageMemoryBarrier2-dstStageMask-03930
                    if stages.intersects(PipelineStages::GEOMETRY_SHADER)
                        && !device.enabled_features().geometry_shader
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::GEOMETRY_SHADER`",
                            requires_one_of: RequiresOneOf {
                                features: &["geometry_shader"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkMemoryBarrier2-srcStageMask-03930
                    // VUID-VkMemoryBarrier2-dstStageMask-03930
                    // VUID-VkBufferMemoryBarrier2-srcStageMask-03930
                    // VUID-VkBufferMemoryBarrier2-dstStageMask-03930
                    // VUID-VkImageMemoryBarrier2-srcStageMask-03930
                    // VUID-VkImageMemoryBarrier2-dstStageMask-03930
                    if stages.intersects(
                        PipelineStages::TESSELLATION_CONTROL_SHADER
                            | PipelineStages::TESSELLATION_EVALUATION_SHADER,
                    ) && !device.enabled_features().tessellation_shader
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                            `PipelineStages::TESSELLATION_EVALUATION_SHADER`",
                            requires_one_of: RequiresOneOf {
                                features: &["tessellation_shader"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkMemoryBarrier2-srcStageMask-03931
                    // VUID-VkMemoryBarrier2-dstStageMask-03931
                    // VUID-VkBufferMemoryBarrier2-srcStageMask-03931
                    // VUID-VkBufferMemoryBarrier2-dstStageMask-03931
                    // VUID-VImagekMemoryBarrier2-srcStageMask-03931
                    // VUID-VkImageMemoryBarrier2-dstStageMask-03931
                    if stages.intersects(PipelineStages::CONDITIONAL_RENDERING)
                        && !device.enabled_features().conditional_rendering
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::CONDITIONAL_RENDERING`",
                            requires_one_of: RequiresOneOf {
                                features: &["conditional_rendering"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkMemoryBarrier2-srcStageMask-03932
                    // VUID-VkMemoryBarrier2-dstStageMask-03932
                    // VUID-VkBufferMemoryBarrier2-srcStageMask-03932
                    // VUID-VkBufferMemoryBarrier2-dstStageMask-03932
                    // VUID-VkImageMemoryBarrier2-srcStageMask-03932
                    // VUID-VkImageMemoryBarrier2-dstStageMask-03932
                    if stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS)
                        && !device.enabled_features().fragment_density_map
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`",
                            requires_one_of: RequiresOneOf {
                                features: &["fragment_density_map"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkMemoryBarrier2-srcStageMask-03933
                    // VUID-VkMemoryBarrier2-dstStageMask-03933
                    // VUID-VkBufferMemoryBarrier2-srcStageMask-03933
                    // VUID-VkBufferMemoryBarrier2-dstStageMask-03933
                    // VUID-VkImageMemoryBarrier2-srcStageMask-03933
                    // VUID-VkImageMemoryBarrier2-dstStageMask-03933
                    if stages.intersects(PipelineStages::TRANSFORM_FEEDBACK)
                        && !device.enabled_features().transform_feedback
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::TRANSFORM_FEEDBACK`",
                            requires_one_of: RequiresOneOf {
                                features: &["transform_feedback"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkMemoryBarrier2-srcStageMask-03934
                    // VUID-VkMemoryBarrier2-dstStageMask-03934
                    // VUID-VkBufferMemoryBarrier2-srcStageMask-03934
                    // VUID-VkBufferMemoryBarrier2-dstStageMask-03934
                    // VUID-VkImageMemoryBarrier2-srcStageMask-03934
                    // VUID-VkImageMemoryBarrier2-dstStageMask-03934
                    if stages.intersects(PipelineStages::MESH_SHADER)
                        && !device.enabled_features().mesh_shader
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::MESH_SHADER`",
                            requires_one_of: RequiresOneOf {
                                features: &["mesh_shader"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkMemoryBarrier2-srcStageMask-03935
                    // VUID-VkMemoryBarrier2-dstStageMask-03935
                    // VUID-VkBufferMemoryBarrier2-srcStageMask-03935
                    // VUID-VkBufferMemoryBarrier2-dstStageMask-03935
                    // VUID-VkImageMemoryBarrier2-srcStageMask-03935
                    // VUID-VkImageMemoryBarrier2-dstStageMask-03935
                    if stages.intersects(PipelineStages::TASK_SHADER)
                        && !device.enabled_features().task_shader
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::TASK_SHADER`",
                            requires_one_of: RequiresOneOf {
                                features: &["task_shader"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkMemoryBarrier2-shadingRateImage-07316
                    // VUID-VkMemoryBarrier2-shadingRateImage-07316
                    // VUID-VkBufferMemoryBarrier2-shadingRateImage-07316
                    // VUID-VkBufferMemoryBarrier2-shadingRateImage-07316
                    // VUID-VkImageMemoryBarrier2-shadingRateImage-07316
                    // VUID-VkImageMemoryBarrier2-shadingRateImage-07316
                    if stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT)
                        && !(device.enabled_features().attachment_fragment_shading_rate
                            || device.enabled_features().shading_rate_image)
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`",
                            requires_one_of: RequiresOneOf {
                                features: &[
                                    "attachment_fragment_shading_rate",
                                    "shading_rate_image",
                                ],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkMemoryBarrier2-srcStageMask-04957
                    // VUID-VkMemoryBarrier2-dstStageMask-04957
                    // VUID-VkBufferMemoryBarrier2-srcStageMask-04957
                    // VUID-VkBufferMemoryBarrier2-dstStageMask-04957
                    // VUID-VkImageMemoryBarrier2-srcStageMask-04957
                    // VUID-VkImageMemoryBarrier2-dstStageMask-04957
                    if stages.intersects(PipelineStages::SUBPASS_SHADING)
                        && !device.enabled_features().subpass_shading
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::SUBPASS_SHADING`",
                            requires_one_of: RequiresOneOf {
                                features: &["subpass_shading"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkMemoryBarrier2-srcStageMask-04995
                    // VUID-VkMemoryBarrier2-dstStageMask-04995
                    // VUID-VkBufferMemoryBarrier2-srcStageMask-04995
                    // VUID-VkBufferMemoryBarrier2-dstStageMask-04995
                    // VUID-VkImageMemoryBarrier2-srcStageMask-04995
                    // VUID-VkImageMemoryBarrier2-dstStageMask-04995
                    if stages.intersects(PipelineStages::INVOCATION_MASK)
                        && !device.enabled_features().invocation_mask
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            contains `PipelineStages::INVOCATION_MASK`",
                            requires_one_of: RequiresOneOf {
                                features: &["invocation_mask"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-vkCmdWaitEvents-srcStageMask-03937
                    // VUID-vkCmdWaitEvents-dstStageMask-03937
                    if stages.is_empty() && !device.enabled_features().synchronization2 {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "One of `dependency_info.memory_barriers`, \
                            `dependency_info.buffer_memory_barriers` or \
                            `dependency_info.image_memory_barriers` has an element where `stages` \
                            is empty",
                            requires_one_of: RequiresOneOf {
                                features: &["synchronization2"],
                                ..Default::default()
                            },
                        });
                    }

                    // A bit of a ridiculous number of VUIDs...

                    // VUID-VkMemoryBarrier2-srcAccessMask-03900
                    // ..
                    // VUID-VkMemoryBarrier2-srcAccessMask-07458

                    // VUID-VkMemoryBarrier2-dstAccessMask-03900
                    // ..
                    // VUID-VkMemoryBarrier2-dstAccessMask-07458

                    // VUID-VkBufferMemoryBarrier2-srcAccessMask-03900
                    // ..
                    // VUID-VkBufferMemoryBarrier2-srcAccessMask-07458

                    // VUID-VkBufferMemoryBarrier2-dstAccessMask-03900
                    // ..
                    // VUID-VkBufferMemoryBarrier2-dstAccessMask-07458

                    // VUID-VkImageMemoryBarrier2-srcAccessMask-03900
                    // ..
                    // VUID-VkImageMemoryBarrier2-srcAccessMask-07458

                    // VUID-VkImageMemoryBarrier2-dstAccessMask-03900
                    // ..
                    // VUID-VkImageMemoryBarrier2-dstAccessMask-07458

                    if !AccessFlags::from(stages).contains(access) {
                        match ty {
                            'm' => {
                                return Err(
                                    SynchronizationError::MemoryBarrierAccessNotSupportedByStages {
                                        barrier_index,
                                    },
                                )
                            }
                            'b' => return Err(
                                SynchronizationError::BufferMemoryBarrierAccessNotSupportedByStages {
                                    barrier_index,
                                },
                            ),
                            'i' => return Err(
                                SynchronizationError::ImageMemoryBarrierAccessNotSupportedByStages {
                                    barrier_index,
                                },
                            ),
                            _ => unreachable!(),
                        }
                    }
                }

                // VUID-VkMemoryBarrier2-srcAccessMask-06256
                // VUID-VkBufferMemoryBarrier2-srcAccessMask-06256
                // VUID-VkImageMemoryBarrier2-srcAccessMask-06256
                if !device.enabled_features().ray_query
                    && src_access.intersects(AccessFlags::ACCELERATION_STRUCTURE_READ)
                    && src_stages.intersects(
                        PipelineStages::VERTEX_SHADER
                            | PipelineStages::TESSELLATION_CONTROL_SHADER
                            | PipelineStages::TESSELLATION_EVALUATION_SHADER
                            | PipelineStages::GEOMETRY_SHADER
                            | PipelineStages::FRAGMENT_SHADER
                            | PipelineStages::COMPUTE_SHADER
                            | PipelineStages::PRE_RASTERIZATION_SHADERS
                            | PipelineStages::TASK_SHADER
                            | PipelineStages::MESH_SHADER,
                    )
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "One of `dependency_info.memory_barriers`, \
                        `dependency_info.buffer_memory_barriers` or \
                        `dependency_info.image_memory_barriers` has an element where \
                        `src_access` contains `ACCELERATION_STRUCTURE_READ`, and \
                        `src_stages` contains a shader stage other than `RAY_TRACING_SHADER`",
                        requires_one_of: RequiresOneOf {
                            features: &["ray_query"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-vkCmdWaitEvents2-dependencyFlags-03844
                if self.builder_state.render_pass.is_some()
                    && src_stages.intersects(PipelineStages::HOST)
                {
                    todo!()
                }

                Ok(())
            };

            let check_queue_family_ownership_transfer =
                |ty: char,
                 barrier_index: usize,
                 src_stages: PipelineStages,
                 dst_stages: PipelineStages,
                 queue_family_ownership_transfer: Option<QueueFamilyOwnershipTransfer>,
                 sharing: &Sharing<_>|
                 -> Result<(), SynchronizationError> {
                    if let Some(transfer) = queue_family_ownership_transfer {
                        // VUID?
                        transfer.validate_device(device)?;

                        // VUID-VkBufferMemoryBarrier2-srcQueueFamilyIndex-04087
                        // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04070
                        // Ensured by the definition of `QueueFamilyOwnershipTransfer`.

                        // VUID-VkBufferMemoryBarrier2-buffer-04088
                        // VUID-VkImageMemoryBarrier2-image-04071
                        // Ensured by the definition of `QueueFamilyOwnershipTransfer`.

                        let queue_family_count =
                            device.physical_device().queue_family_properties().len() as u32;

                        let provided_queue_family_index = match (sharing, transfer) {
                            (
                                Sharing::Exclusive,
                                QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                                    src_index,
                                    dst_index,
                                },
                            ) => Some(max(src_index, dst_index)),
                            (
                                Sharing::Exclusive,
                                QueueFamilyOwnershipTransfer::ExclusiveToExternal { src_index }
                                | QueueFamilyOwnershipTransfer::ExclusiveToForeign { src_index },
                            ) => Some(src_index),
                            (
                                Sharing::Exclusive,
                                QueueFamilyOwnershipTransfer::ExclusiveFromExternal { dst_index }
                                | QueueFamilyOwnershipTransfer::ExclusiveFromForeign { dst_index },
                            ) => Some(dst_index),
                            (
                                Sharing::Concurrent(_),
                                QueueFamilyOwnershipTransfer::ConcurrentToExternal
                                | QueueFamilyOwnershipTransfer::ConcurrentFromExternal
                                | QueueFamilyOwnershipTransfer::ConcurrentToForeign
                                | QueueFamilyOwnershipTransfer::ConcurrentFromForeign,
                            ) => None,
                            _ => match ty {
                                'b' => return Err(SynchronizationError::BufferMemoryBarrierOwnershipTransferSharingMismatch {
                                    barrier_index,
                                }),
                                'i' => return Err(SynchronizationError::ImageMemoryBarrierOwnershipTransferSharingMismatch {
                                    barrier_index,
                                }),
                                _ => unreachable!(),
                            },
                        }.filter(|&index| index >= queue_family_count);

                        // VUID-VkBufferMemoryBarrier2-buffer-04089
                        // VUID-VkImageMemoryBarrier2-image-04072

                        if let Some(provided_queue_family_index) = provided_queue_family_index {
                            match ty {
                                'b' => return Err(SynchronizationError::BufferMemoryBarrierOwnershipTransferIndexOutOfRange {
                                    barrier_index,
                                    provided_queue_family_index,
                                    queue_family_count,
                                }),
                                'i' => return Err(SynchronizationError::ImageMemoryBarrierOwnershipTransferIndexOutOfRange {
                                    barrier_index,
                                    provided_queue_family_index,
                                    queue_family_count,
                                }),
                                _ => unreachable!(),
                            }
                        }

                        // VUID-VkBufferMemoryBarrier2-srcStageMask-03851
                        // VUID-VkImageMemoryBarrier2-srcStageMask-03854
                        if src_stages.intersects(PipelineStages::HOST)
                            || dst_stages.intersects(PipelineStages::HOST)
                        {
                            match ty {
                                'b' => return Err(SynchronizationError::BufferMemoryBarrierOwnershipTransferHostNotAllowed {
                                    barrier_index,
                                }),
                                'i' => return Err(SynchronizationError::ImageMemoryBarrierOwnershipTransferHostForbidden {
                                    barrier_index,
                                }),
                                _ => unreachable!(),
                            }
                        }
                    }

                    Ok(())
                };

            for (barrier_index, barrier) in memory_barriers.iter().enumerate() {
                let &MemoryBarrier {
                    src_stages,
                    src_access,
                    dst_stages,
                    dst_access,
                    _ne: _,
                } = barrier;

                /*
                    Check stages and access
                */

                check_stages_access(
                    'm',
                    barrier_index,
                    src_stages,
                    src_access,
                    dst_stages,
                    dst_access,
                )?;
            }

            for (barrier_index, barrier) in buffer_memory_barriers.iter().enumerate() {
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

                // VUID-VkBufferMemoryBarrier2-buffer-01931
                // Ensured by Buffer type construction.

                /*
                    Check stages and access
                */

                check_stages_access(
                    'b',
                    barrier_index,
                    src_stages,
                    src_access,
                    dst_stages,
                    dst_access,
                )?;

                /*
                    Check queue family transfer
                */

                check_queue_family_ownership_transfer(
                    'b',
                    barrier_index,
                    src_stages,
                    dst_stages,
                    queue_family_ownership_transfer,
                    buffer.sharing(),
                )?;

                /*
                    Check range
                */

                // VUID-VkBufferMemoryBarrier2-size-01188
                assert!(!range.is_empty());

                // VUID-VkBufferMemoryBarrier2-offset-01187
                // VUID-VkBufferMemoryBarrier2-size-01189
                if range.end > buffer.size() {
                    return Err(SynchronizationError::BufferMemoryBarrierOutOfRange {
                        barrier_index,
                        range_end: range.end,
                        buffer_size: buffer.size(),
                    });
                }
            }

            for (barrier_index, barrier) in image_memory_barriers.iter().enumerate() {
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

                // VUID-VkImageMemoryBarrier2-image-01932
                // Ensured by Image type construction.

                /*
                    Check stages and access
                */

                check_stages_access(
                    'i',
                    barrier_index,
                    src_stages,
                    src_access,
                    dst_stages,
                    dst_access,
                )?;

                /*
                    Check layouts
                */

                // VUID-VkImageMemoryBarrier2-oldLayout-parameter
                old_layout.validate_device(device)?;

                // VUID-VkImageMemoryBarrier2-newLayout-parameter
                new_layout.validate_device(device)?;

                // VUID-VkImageMemoryBarrier2-srcStageMask-03855
                if src_stages.intersects(PipelineStages::HOST)
                    && !matches!(
                        old_layout,
                        ImageLayout::Preinitialized | ImageLayout::Undefined | ImageLayout::General
                    )
                {
                    return Err(
                        SynchronizationError::ImageMemoryBarrierOldLayoutFromHostInvalid {
                            barrier_index,
                            old_layout,
                        },
                    );
                }

                // VUID-VkImageMemoryBarrier2-oldLayout-01197
                // Not checked yet, therefore unsafe.

                // VUID-VkImageMemoryBarrier2-newLayout-01198
                if matches!(
                    new_layout,
                    ImageLayout::Undefined | ImageLayout::Preinitialized
                ) {
                    return Err(SynchronizationError::ImageMemoryBarrierNewLayoutInvalid {
                        barrier_index,
                    });
                }

                // VUID-VkImageMemoryBarrier2-attachmentFeedbackLoopLayout-07313
                /*if !device.enabled_features().attachment_feedback_loop_layout
                    && matches!(new_layout, ImageLayout::AttachmentFeedbackLoopOptimal)
                {
                    return Err(SynchronizationError::RequirementNotMet {
                        required_for: "`dependency_info.image_memory_barriers` has an element where \
                            `new_layout` is `AttachmentFeedbackLoopOptimal`",
                        requires_one_of: RequiresOneOf {
                            features: &["attachment_feedback_loop_layout"],
                            ..Default::default()
                        },
                    });
                }*/

                for layout in [old_layout, new_layout] {
                    // VUID-VkImageMemoryBarrier2-synchronization2-06911
                    /*if !device.enabled_features().synchronization2
                        && matches!(
                            layout,
                            ImageLayout::AttachmentOptimal | ImageLayout::ReadOnlyOptimal
                        )
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "`dependency_info.image_memory_barriers` has an element \
                                where `old_layout` or `new_layout` is `AttachmentOptimal` or \
                                `ReadOnlyOptimal`",
                            requires_one_of: RequiresOneOf {
                                features: &["synchronization2"],
                                ..Default::default()
                            },
                        });
                    }*/

                    // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-07006
                    /*if layout == ImageLayout::AttachmentFeedbackLoopOptimal {
                        if !image.usage().intersects(
                            ImageUsage::COLOR_ATTACHMENT | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                        ) {
                            return Err(
                                SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                                    barrier_index,
                                    layout,
                                    requires_one_of_usage: ImageUsage::COLOR_ATTACHMENT
                                        | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                                },
                            );
                        }

                        if !image
                            .usage()
                            .intersects(ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED)
                        {
                            return Err(
                                SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                                    barrier_index,
                                    layout,
                                    requires_one_of_usage: ImageUsage::INPUT_ATTACHMENT
                                        | ImageUsage::SAMPLED,
                                },
                            );
                        }

                        if !image
                            .usage()
                            .intersects(ImageUsage::ATTACHMENT_FEEDBACK_LOOP)
                        {
                            return Err(
                                SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                                    barrier_index,
                                    layout,
                                    requires_one_of_usage: ImageUsage::ATTACHMENT_FEEDBACK_LOOP,
                                },
                            );
                        }
                    }*/

                    let requires_one_of_usage = match layout {
                        // VUID-VkImageMemoryBarrier2-oldLayout-01208
                        ImageLayout::ColorAttachmentOptimal => ImageUsage::COLOR_ATTACHMENT,

                        // VUID-VkImageMemoryBarrier2-oldLayout-01209
                        ImageLayout::DepthStencilAttachmentOptimal => {
                            ImageUsage::DEPTH_STENCIL_ATTACHMENT
                        }

                        // VUID-VkImageMemoryBarrier2-oldLayout-01210
                        ImageLayout::DepthStencilReadOnlyOptimal => {
                            ImageUsage::DEPTH_STENCIL_ATTACHMENT
                        }

                        // VUID-VkImageMemoryBarrier2-oldLayout-01211
                        ImageLayout::ShaderReadOnlyOptimal => {
                            ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT
                        }

                        // VUID-VkImageMemoryBarrier2-oldLayout-01212
                        ImageLayout::TransferSrcOptimal => ImageUsage::TRANSFER_SRC,

                        // VUID-VkImageMemoryBarrier2-oldLayout-01213
                        ImageLayout::TransferDstOptimal => ImageUsage::TRANSFER_DST,

                        // VUID-VkImageMemoryBarrier2-oldLayout-01658
                        ImageLayout::DepthReadOnlyStencilAttachmentOptimal => {
                            ImageUsage::DEPTH_STENCIL_ATTACHMENT
                        }

                        // VUID-VkImageMemoryBarrier2-oldLayout-01659
                        ImageLayout::DepthAttachmentStencilReadOnlyOptimal => {
                            ImageUsage::DEPTH_STENCIL_ATTACHMENT
                        }

                        /*
                        // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04065
                        ImageLayout::DepthReadOnlyOptimal => {
                            ImageUsage::DEPTH_STENCIL_ATTACHMENT
                                | ImageUsage::SAMPLED
                                | ImageUsage::INPUT_ATTACHMENT
                        }

                        // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04066
                        ImageLayout::DepthAttachmentOptimal => ImageUsage::DEPTH_STENCIL_ATTACHMENT,

                        // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04067
                        ImageLayout::StencilReadOnlyOptimal => {
                            ImageUsage::DEPTH_STENCIL_ATTACHMENT
                                | ImageUsage::SAMPLED
                                | ImageUsage::INPUT_ATTACHMENT
                        }

                        // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04068
                        ImageLayout::StencilAttachmentOptimal => ImageUsage::DEPTH_STENCIL_ATTACHMENT,

                        // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-03938
                        ImageLayout::AttachmentOptimal => {
                            ImageUsage::COLOR_ATTACHMENT | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                        }

                        // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-03939
                        ImageLayout::ReadOnlyOptimal => {
                            ImageUsage::DEPTH_STENCIL_ATTACHMENT
                                | ImageUsage::SAMPLED
                                | ImageUsage::INPUT_ATTACHMENT
                        }

                        // VUID-VkImageMemoryBarrier2-oldLayout-02088
                        ImageLayout::FragmentShadingRateAttachmentOptimal => {
                            ImageUsage::FRAGMENT_SHADING_RATE_ATTACHMENT
                        }
                         */
                        _ => continue,
                    };

                    if !image.usage().intersects(requires_one_of_usage) {
                        return Err(
                            SynchronizationError::ImageMemoryBarrierImageMissingUsageForLayout {
                                barrier_index,
                                layout,
                                requires_one_of_usage,
                            },
                        );
                    }
                }

                /*
                    Check queue family tansfer
                */

                check_queue_family_ownership_transfer(
                    'i',
                    barrier_index,
                    src_stages,
                    dst_stages,
                    queue_family_ownership_transfer,
                    image.sharing(),
                )?;

                /*
                    Check subresource range
                */

                // VUID-VkImageSubresourceRange-aspectMask-requiredbitmask
                assert!(!subresource_range.aspects.is_empty());

                // VUID-VkImageSubresourceRange-aspectMask-parameter
                subresource_range.aspects.validate_device(device)?;

                let image_aspects = image.format().unwrap().aspects();

                // VUID-VkImageMemoryBarrier2-image-01673
                // VUID-VkImageMemoryBarrier2-image-03319
                if image_aspects.contains(subresource_range.aspects) {
                    return Err(SynchronizationError::ImageMemoryBarrierAspectsNotAllowed {
                        barrier_index,
                        aspects: subresource_range.aspects - image_aspects,
                    });
                }

                if image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
                    // VUID-VkImageMemoryBarrier2-image-03320
                    if !device.enabled_features().separate_depth_stencil_layouts
                        && image_aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                        && !subresource_range
                            .aspects
                            .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                    {
                        return Err(SynchronizationError::RequirementNotMet {
                            required_for: "`dependency_info.image_memory_barriers` has an element \
                            where `image` has both a depth and a stencil aspect, and \
                            `subresource_range.aspects` does not contain both aspects",
                            requires_one_of: RequiresOneOf {
                                features: &["separate_depth_stencil_layouts"],
                                ..Default::default()
                            },
                        });
                    }
                } else {
                    // VUID-VkImageMemoryBarrier2-image-01671
                    if !image.flags().intersects(ImageCreateFlags::DISJOINT)
                        && subresource_range.aspects != ImageAspects::COLOR
                    {
                        return Err(SynchronizationError::ImageMemoryBarrierAspectsNotAllowed {
                            barrier_index,
                            aspects: subresource_range.aspects - ImageAspects::COLOR,
                        });
                    }
                }

                // VUID-VkImageSubresourceRange-levelCount-01720
                assert!(!subresource_range.mip_levels.is_empty());

                // VUID-VkImageMemoryBarrier2-subresourceRange-01486
                // VUID-VkImageMemoryBarrier2-subresourceRange-01724
                if subresource_range.mip_levels.end > image.mip_levels() {
                    return Err(
                        SynchronizationError::ImageMemoryBarrierMipLevelsOutOfRange {
                            barrier_index,
                            mip_levels_range_end: subresource_range.mip_levels.end,
                            image_mip_levels: image.mip_levels(),
                        },
                    );
                }

                // VUID-VkImageSubresourceRange-layerCount-01721
                assert!(!subresource_range.array_layers.is_empty());

                // VUID-VkImageMemoryBarrier2-subresourceRange-01488
                // VUID-VkImageMemoryBarrier2-subresourceRange-01725
                if subresource_range.array_layers.end > image.dimensions().array_layers() {
                    return Err(
                        SynchronizationError::ImageMemoryBarrierArrayLayersOutOfRange {
                            barrier_index,
                            array_layers_range_end: subresource_range.array_layers.end,
                            image_array_layers: image.dimensions().array_layers(),
                        },
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wait_events_unchecked(
        &mut self,
        events: impl IntoIterator<Item = (Arc<Event>, DependencyInfo)>,
    ) -> &mut Self {
        let events: SmallVec<[(Arc<Event>, DependencyInfo); 4]> = events.into_iter().collect();
        let fns = self.device().fns();

        // VUID-vkCmdWaitEvents2-pEvents-03837
        // Ensured by using `vkCmdSetEvent2` and `vkCmdWaitEvents2` under the exact same
        // conditions, i.e. when `synchronization2` is enabled.

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
                    dependency_flags,
                    ref memory_barriers,
                    ref buffer_memory_barriers,
                    ref image_memory_barriers,
                    _ne: _,
                } = dependency_info;

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
                debug_assert!(self.device().enabled_extensions().khr_synchronization2);
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

                let DependencyInfo {
                    dependency_flags: _,
                    memory_barriers,
                    buffer_memory_barriers,
                    image_memory_barriers,
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

        self.resources
            .extend(events.into_iter().map(|(event, _)| Box::new(event) as _));

        // TODO: sync state update

        self.next_command_index += 1;
        self
    }

    /// Resets an [`Event`] back to the unsignaled state.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for `event` against any previous
    ///   [`set_event`] or [`wait_events`] command.
    ///
    /// [`set_event`]: Self::set_event
    /// [`wait_events`]: Self::wait_events
    #[inline]
    pub unsafe fn reset_event(
        &mut self,
        event: Arc<Event>,
        stages: PipelineStages,
    ) -> Result<&mut Self, SynchronizationError> {
        self.validate_reset_event(&event, stages)?;

        unsafe { Ok(self.reset_event_unchecked(event, stages)) }
    }

    fn validate_reset_event(
        &self,
        event: &Event,
        stages: PipelineStages,
    ) -> Result<(), SynchronizationError> {
        // VUID-vkCmdResetEvent2-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(SynchronizationError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdResetEvent2-commandBuffer-03833
        // TODO:

        let device = self.device();
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdResetEvent2-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(SynchronizationError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdResetEvent2-commonparent
        assert_eq!(device, event.device());

        // VUID-vkCmdResetEvent2-stageMask-parameter
        stages.validate_device(device)?;

        // VUID-vkCmdResetEvent2-synchronization2-03829
        if !device.enabled_features().synchronization2 {
            if stages.is_2() {
                return Err(SynchronizationError::RequirementNotMet {
                    required_for: "`stages` contains flags from `VkPipelineStageFlagBits2`",
                    requires_one_of: RequiresOneOf {
                        features: &["synchronization2"],
                        ..Default::default()
                    },
                });
            }
        }

        // VUID-vkCmdResetEvent2-stageMask-03929
        if stages.intersects(PipelineStages::GEOMETRY_SHADER)
            && !device.enabled_features().geometry_shader
        {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` contains `PipelineStages::GEOMETRY_SHADER`",
                requires_one_of: RequiresOneOf {
                    features: &["geometry_shader"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent2-stageMask-03930
        if stages.intersects(
            PipelineStages::TESSELLATION_CONTROL_SHADER
                | PipelineStages::TESSELLATION_EVALUATION_SHADER,
        ) && !device.enabled_features().tessellation_shader
        {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                    `PipelineStages::TESSELLATION_EVALUATION_SHADER`",
                requires_one_of: RequiresOneOf {
                    features: &["tessellation_shader"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent2-stageMask-03931
        if stages.intersects(PipelineStages::CONDITIONAL_RENDERING)
            && !device.enabled_features().conditional_rendering
        {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` contains `PipelineStages::CONDITIONAL_RENDERING`",
                requires_one_of: RequiresOneOf {
                    features: &["conditional_rendering"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent2-stageMask-03932
        if stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS)
            && !device.enabled_features().fragment_density_map
        {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`",
                requires_one_of: RequiresOneOf {
                    features: &["fragment_density_map"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent2-stageMask-03933
        if stages.intersects(PipelineStages::TRANSFORM_FEEDBACK)
            && !device.enabled_features().transform_feedback
        {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` contains `PipelineStages::TRANSFORM_FEEDBACK`",
                requires_one_of: RequiresOneOf {
                    features: &["transform_feedback"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent2-stageMask-03934
        if stages.intersects(PipelineStages::MESH_SHADER) && !device.enabled_features().mesh_shader
        {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` contains `PipelineStages::MESH_SHADER`",
                requires_one_of: RequiresOneOf {
                    features: &["mesh_shader"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent2-stageMask-03935
        if stages.intersects(PipelineStages::TASK_SHADER) && !device.enabled_features().task_shader
        {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` contains `PipelineStages::TASK_SHADER`",
                requires_one_of: RequiresOneOf {
                    features: &["task_shader"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent2-shadingRateImage-07316
        if stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT)
            && !(device.enabled_features().attachment_fragment_shading_rate
                || device.enabled_features().shading_rate_image)
        {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` contains \
                    `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`",
                requires_one_of: RequiresOneOf {
                    features: &["attachment_fragment_shading_rate", "shading_rate_image"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent2-stageMask-04957
        if stages.intersects(PipelineStages::SUBPASS_SHADING)
            && !device.enabled_features().subpass_shading
        {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` contains `PipelineStages::SUBPASS_SHADING`",
                requires_one_of: RequiresOneOf {
                    features: &["subpass_shading"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent2-stageMask-04995
        if stages.intersects(PipelineStages::INVOCATION_MASK)
            && !device.enabled_features().invocation_mask
        {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` contains `PipelineStages::INVOCATION_MASK`",
                requires_one_of: RequiresOneOf {
                    features: &["invocation_mask"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent-stageMask-03937
        if stages.is_empty() && !device.enabled_features().synchronization2 {
            return Err(SynchronizationError::RequirementNotMet {
                required_for: "`stages` is empty",
                requires_one_of: RequiresOneOf {
                    features: &["synchronization2"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdResetEvent2-stageMask-03830
        if stages.intersects(PipelineStages::HOST) {
            todo!()
        }

        // VUID-vkCmdResetEvent2-event-03831
        // VUID-vkCmdResetEvent2-event-03832
        // TODO:

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_event_unchecked(
        &mut self,
        event: Arc<Event>,
        stages: PipelineStages,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_reset_event2)(self.handle(), event.handle(), stages.into());
            } else {
                debug_assert!(self.device().enabled_extensions().khr_synchronization2);
                (fns.khr_synchronization2.cmd_reset_event2_khr)(
                    self.handle(),
                    event.handle(),
                    stages.into(),
                );
            }
        } else {
            (fns.v1_0.cmd_reset_event)(self.handle(), event.handle(), stages.into());
        }

        self.resources.push(Box::new(event));

        // TODO: sync state update

        self.next_command_index += 1;
        self
    }
}

/// Error that can happen when recording a synchronization command.
#[derive(Clone, Debug)]
pub enum SynchronizationError {
    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// One or more accesses of a buffer memory barrier are not supported by the corresponding
    /// pipeline stages.
    BufferMemoryBarrierAccessNotSupportedByStages { barrier_index: usize },

    /// Buffer memory barriers are forbidden inside a render pass instance.
    BufferMemoryBarrierForbiddenInsideRenderPass,

    /// The end of `range` of a buffer memory barrier is greater than the size of `buffer`.
    BufferMemoryBarrierOutOfRange {
        barrier_index: usize,
        range_end: DeviceSize,
        buffer_size: DeviceSize,
    },

    /// A buffer memory barrier contains a queue family ownership transfer, but either the
    /// `src_stages` or `dst_stages` contain [`HOST`].
    ///
    /// [`HOST`]: crate::sync::PipelineStages::HOST
    BufferMemoryBarrierOwnershipTransferHostNotAllowed { barrier_index: usize },

    /// The provided `src_index` or `dst_index` in the queue family ownership transfer of a
    /// buffer memory barrier is not less than the number of queue families in the physical device.
    BufferMemoryBarrierOwnershipTransferIndexOutOfRange {
        barrier_index: usize,
        provided_queue_family_index: u32,
        queue_family_count: u32,
    },

    /// The provided `queue_family_ownership_transfer` value of a buffer memory barrier does not
    /// match the sharing mode of `buffer`.
    BufferMemoryBarrierOwnershipTransferSharingMismatch { barrier_index: usize },

    /// One or more pipeline stages of a buffer memory barrier are not supported by the queue
    /// family of the command buffer.
    BufferMemoryBarrierStageNotSupported { barrier_index: usize },

    /// A render pass instance is not active, and the `VIEW_LOCAL` dependency flag was provided.
    DependencyFlagsViewLocalNotAllowed,

    /// Operation forbidden inside a render pass.
    ForbiddenInsideRenderPass,

    /// Operation forbidden inside a render pass instance that was begun with `begin_rendering`.
    ForbiddenWithBeginRendering,

    /// One or more accesses of an image memory barrier are not supported by the corresponding
    /// pipeline stages.
    ImageMemoryBarrierAccessNotSupportedByStages { barrier_index: usize },

    /// The end of the range of array layers of the subresource range of an image memory barrier
    /// is greater than the number of array layers in the image.
    ImageMemoryBarrierArrayLayersOutOfRange {
        barrier_index: usize,
        array_layers_range_end: u32,
        image_array_layers: u32,
    },

    /// The aspects of the subresource range of an image memory barrier contain aspects that are
    /// not present in the image, or that are not allowed.
    ImageMemoryBarrierAspectsNotAllowed {
        barrier_index: usize,
        aspects: ImageAspects,
    },

    /// For the `old_layout` or `new_layout` of an image memory barrier, `image` does not have a
    /// usage that is required.
    ImageMemoryBarrierImageMissingUsageForLayout {
        barrier_index: usize,
        layout: ImageLayout,
        requires_one_of_usage: ImageUsage,
    },

    /// An image memory barrier contains an image layout transition, but a render pass
    /// instance is active.
    ImageMemoryBarrierLayoutTransitionForbiddenInsideRenderPass { barrier_index: usize },

    /// The end of the range of mip levels of the subresource range of an image memory barrier
    /// is greater than the number of mip levels in the image.
    ImageMemoryBarrierMipLevelsOutOfRange {
        barrier_index: usize,
        mip_levels_range_end: u32,
        image_mip_levels: u32,
    },

    /// The `new_layout` of an image memory barrier is `Undefined` or `Preinitialized`.
    ImageMemoryBarrierNewLayoutInvalid { barrier_index: usize },

    /// A render pass instance is active, and the image of an image memory barrier is not a color
    /// or depth/stencil attachment of the current subpass.
    ImageMemoryBarrierNotColorDepthStencilAttachment { barrier_index: usize },

    /// A render pass instance is active, and the image of an image memory barrier is not an input
    /// attachment of the current subpass.
    ImageMemoryBarrierNotInputAttachment { barrier_index: usize },

    /// The `src_stages` of an image memory barrier contains [`HOST`], but `old_layout` is not
    /// `Preinitialized`, `Undefined` or `General`.
    ///
    /// [`HOST`]: crate::sync::PipelineStages::HOST
    ImageMemoryBarrierOldLayoutFromHostInvalid {
        barrier_index: usize,
        old_layout: ImageLayout,
    },

    /// An image memory barrier contains a queue family ownership transfer, but a render pass
    /// instance is active.
    ImageMemoryBarrierOwnershipTransferForbiddenInsideRenderPass { barrier_index: usize },

    /// An image memory barrier contains a queue family ownership transfer, but either the
    /// `src_stages` or `dst_stages` contain [`HOST`].
    ///
    /// [`HOST`]: crate::sync::PipelineStages::HOST
    ImageMemoryBarrierOwnershipTransferHostForbidden { barrier_index: usize },

    /// The provided `src_index` or `dst_index` in the queue family ownership transfer of an
    /// image memory barrier is not less than the number of queue families in the physical device.
    ImageMemoryBarrierOwnershipTransferIndexOutOfRange {
        barrier_index: usize,
        provided_queue_family_index: u32,
        queue_family_count: u32,
    },

    /// The provided `queue_family_ownership_transfer` value of an image memory barrier does not
    /// match the sharing mode of `image`.
    ImageMemoryBarrierOwnershipTransferSharingMismatch { barrier_index: usize },

    /// One or more pipeline stages of an image memory barrier are not supported by the queue
    /// family of the command buffer.
    ImageMemoryBarrierStageNotSupported { barrier_index: usize },

    /// One or more accesses of a memory barrier are not supported by the corresponding
    /// pipeline stages.
    MemoryBarrierAccessNotSupportedByStages { barrier_index: usize },

    /// A render pass instance is active, but the render pass does not have a subpass
    /// self-dependency for the current subpass that is a superset of the barriers.
    MemoryBarrierNoMatchingSubpassSelfDependency,

    /// One or more pipeline stages of a memory barrier are not supported by the queue
    /// family of the command buffer.
    MemoryBarrierStageNotSupported { barrier_index: usize },

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,
}

impl Error for SynchronizationError {}

impl Display for SynchronizationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),

            Self::BufferMemoryBarrierAccessNotSupportedByStages { barrier_index } => write!(
                f,
                "one or more accesses of buffer memory barrier {} are not supported by the \
                corresponding pipeline stages",
                barrier_index,
            ),
            Self::BufferMemoryBarrierForbiddenInsideRenderPass => write!(
                f,
                "buffer memory barriers are forbidden inside a render pass instance",
            ),
            Self::BufferMemoryBarrierOutOfRange {
                barrier_index,
                range_end,
                buffer_size,
            } => write!(
                f,
                "the end of `range` ({}) of buffer memory barrier {} is greater than the size of \
                `buffer` ({})",
                range_end, barrier_index, buffer_size,
            ),
            Self::BufferMemoryBarrierOwnershipTransferHostNotAllowed { barrier_index } => write!(
                f,
                "buffer memory barrier {} contains a queue family ownership transfer, but either \
                the `src_stages` or `dst_stages` contain `HOST`",
                barrier_index,
            ),
            Self::BufferMemoryBarrierOwnershipTransferIndexOutOfRange {
                barrier_index,
                provided_queue_family_index,
                queue_family_count,
            } => write!(
                f,
                "the provided `src_index` or `dst_index` ({}) in the queue family ownership \
                transfer of buffer memory barrier {} is not less than the number of queue \
                families in the physical device ({})",
                provided_queue_family_index, barrier_index, queue_family_count,
            ),
            Self::BufferMemoryBarrierOwnershipTransferSharingMismatch { barrier_index } => write!(
                f,
                "the provided `queue_family_ownership_transfer` value of buffer memory barrier {} \
                does not match the sharing mode of `buffer`",
                barrier_index,
            ),
            Self::BufferMemoryBarrierStageNotSupported { barrier_index } => write!(
                f,
                "one or more pipeline stages of buffer memory barrier {} are not supported by the \
                queue family of the command buffer",
                barrier_index,
            ),
            Self::DependencyFlagsViewLocalNotAllowed => write!(
                f,
                "a render pass instance is not active, and the `VIEW_LOCAL` dependency flag was \
                provided",
            ),
            Self::ForbiddenInsideRenderPass => {
                write!(f, "operation forbidden inside a render pass")
            }
            Self::ForbiddenWithBeginRendering => write!(
                f,
                "operation forbidden inside a render pass instance that was begun with \
                `begin_rendering`",
            ),
            Self::ImageMemoryBarrierAccessNotSupportedByStages { barrier_index } => write!(
                f,
                "one or more accesses of image memory barrier {} are not supported by the \
                corresponding pipeline stages",
                barrier_index,
            ),
            Self::ImageMemoryBarrierArrayLayersOutOfRange {
                barrier_index,
                array_layers_range_end,
                image_array_layers,
            } => write!(
                f,
                "the end of the range of array layers ({}) of the subresource range of image \
                memory barrier {} is greater than the number of array layers in the image ({})",
                array_layers_range_end, barrier_index, image_array_layers,
            ),
            Self::ImageMemoryBarrierAspectsNotAllowed {
                barrier_index,
                aspects,
            } => write!(
                f,
                "the aspects of the subresource range of image memory barrier {} contain aspects \
                that are not present in the image, or that are not allowed ({:?})",
                barrier_index, aspects,
            ),
            Self::ImageMemoryBarrierImageMissingUsageForLayout {
                barrier_index,
                layout,
                requires_one_of_usage,
            } => write!(
                f,
                "for the `old_layout` or `new_layout` ({:?}) of image memory barrier {}, `image` \
                does not have a usage that is required ({}{:?})",
                layout,
                barrier_index,
                if requires_one_of_usage.count() > 1 {
                    "one of "
                } else {
                    ""
                },
                requires_one_of_usage,
            ),
            Self::ImageMemoryBarrierLayoutTransitionForbiddenInsideRenderPass { barrier_index } => {
                write!(
                    f,
                    "image memory barrier {} contains an image layout transition, but a render \
                    pass instance is active",
                    barrier_index,
                )
            }
            Self::ImageMemoryBarrierMipLevelsOutOfRange {
                barrier_index,
                mip_levels_range_end,
                image_mip_levels,
            } => write!(
                f,
                "the end of the range of mip levels ({}) of the subresource range of image \
                memory barrier {} is greater than the number of mip levels in the image ({})",
                mip_levels_range_end, barrier_index, image_mip_levels,
            ),
            Self::ImageMemoryBarrierNewLayoutInvalid { barrier_index } => write!(
                f,
                "the `new_layout` of image memory barrier {} is `Undefined` or `Preinitialized`",
                barrier_index,
            ),
            Self::ImageMemoryBarrierNotColorDepthStencilAttachment { barrier_index } => write!(
                f,
                "a render pass instance is active, and the image of image memory barrier {} is \
                not a color or depth/stencil attachment of the current subpass",
                barrier_index,
            ),
            Self::ImageMemoryBarrierNotInputAttachment { barrier_index } => write!(
                f,
                "a render pass instance is active, and the image of image memory barrier {} is \
                not an input attachment of the current subpass",
                barrier_index,
            ),
            Self::ImageMemoryBarrierOldLayoutFromHostInvalid {
                barrier_index,
                old_layout,
            } => write!(
                f,
                "the `src_stages` of image memory barrier {} contains `HOST`, but `old_layout`
                ({:?}) is not `Preinitialized`, `Undefined` or `General`",
                barrier_index, old_layout,
            ),
            Self::ImageMemoryBarrierOwnershipTransferForbiddenInsideRenderPass {
                barrier_index,
            } => write!(
                f,
                "image memory barrier {} contains a queue family ownership transfer, but a render \
                pass instance is active",
                barrier_index,
            ),
            Self::ImageMemoryBarrierOwnershipTransferHostForbidden { barrier_index } => write!(
                f,
                "image memory barrier {} contains a queue family ownership transfer, but either \
                the `src_stages` or `dst_stages` contain `HOST`",
                barrier_index,
            ),
            Self::ImageMemoryBarrierOwnershipTransferIndexOutOfRange {
                barrier_index,
                provided_queue_family_index,
                queue_family_count,
            } => write!(
                f,
                "the provided `src_index` or `dst_index` ({}) in the queue family ownership \
                transfer of image memory barrier {} is not less than the number of queue
                families in the physical device ({})",
                provided_queue_family_index, barrier_index, queue_family_count,
            ),
            Self::ImageMemoryBarrierOwnershipTransferSharingMismatch { barrier_index } => write!(
                f,
                "the provided `queue_family_ownership_transfer` value of image memory barrier {} \
                does not match the sharing mode of `image`",
                barrier_index,
            ),
            Self::ImageMemoryBarrierStageNotSupported { barrier_index } => write!(
                f,
                "one or more pipeline stages of image memory barrier {} are not supported by the \
                queue family of the command buffer",
                barrier_index,
            ),
            Self::MemoryBarrierAccessNotSupportedByStages { barrier_index } => write!(
                f,
                "one or more accesses of memory barrier {} are not supported by the \
                corresponding pipeline stages",
                barrier_index,
            ),
            Self::MemoryBarrierNoMatchingSubpassSelfDependency => write!(
                f,
                "a render pass instance is active, but the render pass does not have a subpass \
                self-dependency for the current subpass that is a superset of the barriers",
            ),
            Self::MemoryBarrierStageNotSupported { barrier_index } => write!(
                f,
                "one or more pipeline stages of memory barrier {} are not supported by the \
                queue family of the command buffer",
                barrier_index,
            ),
            Self::NotSupportedByQueueFamily => {
                write!(f, "the queue family doesn't allow this operation")
            }
        }
    }
}

impl From<RequirementNotMet> for SynchronizationError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}
