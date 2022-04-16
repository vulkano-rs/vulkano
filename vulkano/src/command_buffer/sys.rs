// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub use super::commands::{
    bind_push::UnsafeCommandBufferBuilderBindVertexBuffer, render_pass::RenderPassBeginInfo,
    secondary::UnsafeCommandBufferBuilderExecuteCommands,
};
use super::{
    pool::UnsafeCommandPoolAlloc, CommandBufferInheritanceInfo, CommandBufferLevel,
    CommandBufferUsage,
};
use crate::{
    check_errors,
    device::{Device, DeviceOwned},
    query::QueryPipelineStatisticFlags,
    OomError, VulkanObject,
};
use std::{ptr, sync::Arc};

/// Command buffer being built.
///
/// # Safety
///
/// - All submitted commands must be valid and follow the requirements of the Vulkan specification.
/// - Any resources used by submitted commands must outlive the returned builder and its created
///   command buffer. They must be protected against data races through manual synchronization.
///
/// > **Note**: Some checks are still made with `debug_assert!`. Do not expect to be able to
/// > submit invalid commands.
#[derive(Debug)]
pub struct UnsafeCommandBufferBuilder {
    pub(super) handle: ash::vk::CommandBuffer,
    pub(super) device: Arc<Device>,
    usage: CommandBufferUsage,
}

impl UnsafeCommandBufferBuilder {
    /// Creates a new builder, for recording commands.
    ///
    /// # Safety
    ///
    /// - `pool_alloc` must outlive the returned builder and its created command buffer.
    /// - `kind` must match how `pool_alloc` was created.
    pub unsafe fn new(
        pool_alloc: &UnsafeCommandPoolAlloc,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<UnsafeCommandBufferBuilder, OomError> {
        let CommandBufferBeginInfo {
            usage,
            inheritance_info,
            _ne: _,
        } = begin_info;

        // VUID-vkBeginCommandBuffer-commandBuffer-00049
        // Can't validate

        // VUID-vkBeginCommandBuffer-commandBuffer-00050
        // Can't validate

        let device = pool_alloc.device().clone();

        // VUID-vkBeginCommandBuffer-commandBuffer-00051
        debug_assert_eq!(
            pool_alloc.level() == CommandBufferLevel::Secondary,
            inheritance_info.is_some()
        );

        {
            // VUID-vkBeginCommandBuffer-commandBuffer-02840
            // Guaranteed by use of enum
            let mut flags = ash::vk::CommandBufferUsageFlags::from(usage);

            let inheritance_info = if let Some(inheritance_info) = &inheritance_info {
                let (render_pass, subpass, framebuffer) =
                    if let Some(render_pass) = &inheritance_info.render_pass {
                        flags |= ash::vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE;

                        // VUID-VkCommandBufferInheritanceInfo-commonparent
                        debug_assert_eq!(render_pass.subpass.render_pass().device(), &device);
                        debug_assert!(render_pass
                            .framebuffer
                            .as_ref()
                            .map_or(true, |fb| fb.device() == &device));

                        (
                            // VUID-VkCommandBufferBeginInfo-flags-00055
                            // VUID-VkCommandBufferBeginInfo-flags-06000
                            render_pass.subpass.render_pass().internal_object(),
                            // VUID-VkCommandBufferBeginInfo-flags-06001
                            // Guaranteed by subpass invariants
                            render_pass.subpass.index(),
                            render_pass
                                .framebuffer
                                .as_ref()
                                .map(|fb| fb.internal_object())
                                .unwrap_or_default(),
                        )
                    } else {
                        Default::default()
                    };

                let (occlusion_query_enable, query_flags) =
                    if let Some(flags) = inheritance_info.occlusion_query {
                        // VUID-VkCommandBufferInheritanceInfo-occlusionQueryEnable-00056
                        debug_assert!(device.enabled_features().inherited_queries);

                        // VUID-VkCommandBufferInheritanceInfo-queryFlags-00057
                        let query_flags = if flags.precise {
                            // VUID-vkBeginCommandBuffer-commandBuffer-00052
                            debug_assert!(device.enabled_features().occlusion_query_precise);

                            ash::vk::QueryControlFlags::PRECISE
                        } else {
                            ash::vk::QueryControlFlags::empty()
                        };
                        (ash::vk::TRUE, query_flags)
                    } else {
                        // VUID-VkCommandBufferInheritanceInfo-queryFlags-02788
                        // VUID-vkBeginCommandBuffer-commandBuffer-00052
                        (ash::vk::FALSE, ash::vk::QueryControlFlags::empty())
                    };

                // VUID-VkCommandBufferInheritanceInfo-pipelineStatistics-02789
                // VUID-VkCommandBufferInheritanceInfo-pipelineStatistics-00058
                debug_assert!(
                    inheritance_info.query_statistics_flags == QueryPipelineStatisticFlags::none()
                        || device.enabled_features().pipeline_statistics_query
                );

                Some(ash::vk::CommandBufferInheritanceInfo {
                    render_pass,
                    subpass,
                    framebuffer,
                    occlusion_query_enable,
                    query_flags,
                    pipeline_statistics: inheritance_info.query_statistics_flags.into(),
                    ..Default::default()
                })
            } else {
                None
            };

            let begin_info = ash::vk::CommandBufferBeginInfo {
                flags,
                p_inheritance_info: inheritance_info.as_ref().map_or(ptr::null(), |info| info),
                ..Default::default()
            };

            let fns = device.fns();

            check_errors(
                fns.v1_0
                    .begin_command_buffer(pool_alloc.internal_object(), &begin_info),
            )?;
        }

        Ok(UnsafeCommandBufferBuilder {
            handle: pool_alloc.internal_object(),
            device,
            usage,
        })
    }

    /// Turns the builder into an actual command buffer.
    #[inline]
    pub fn build(self) -> Result<UnsafeCommandBuffer, OomError> {
        unsafe {
            let fns = self.device.fns();
            check_errors(fns.v1_0.end_command_buffer(self.handle))?;

            Ok(UnsafeCommandBuffer {
                command_buffer: self.handle,
                device: self.device.clone(),
                usage: self.usage,
            })
        }
    }
}

unsafe impl VulkanObject for UnsafeCommandBufferBuilder {
    type Object = ash::vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> ash::vk::CommandBuffer {
        self.handle
    }
}

unsafe impl DeviceOwned for UnsafeCommandBufferBuilder {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// Parameters to begin recording a command buffer.
#[derive(Clone, Debug)]
pub struct CommandBufferBeginInfo {
    /// How the command buffer will be used.
    ///
    /// The default value is [`CommandBufferUsage::MultipleSubmit`].
    pub usage: CommandBufferUsage,

    /// For a secondary command buffer, this must be `Some`, containing the context that will be
    /// inherited from the primary command buffer. For a primary command buffer, this must be
    /// `None`.
    ///
    /// The default value is `None`.
    pub inheritance_info: Option<CommandBufferInheritanceInfo>,

    pub _ne: crate::NonExhaustive,
}

impl Default for CommandBufferBeginInfo {
    #[inline]
    fn default() -> Self {
        Self {
            usage: CommandBufferUsage::MultipleSubmit,
            inheritance_info: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Command buffer that has been built.
///
/// # Safety
///
/// The command buffer must not outlive the command pool that it was created from,
/// nor the resources used by the recorded commands.
pub struct UnsafeCommandBuffer {
    command_buffer: ash::vk::CommandBuffer,
    device: Arc<Device>,
    usage: CommandBufferUsage,
}

impl UnsafeCommandBuffer {
    #[inline]
    pub fn usage(&self) -> CommandBufferUsage {
        self.usage
    }
}

unsafe impl DeviceOwned for UnsafeCommandBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for UnsafeCommandBuffer {
    type Object = ash::vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> ash::vk::CommandBuffer {
        self.command_buffer
    }
}
