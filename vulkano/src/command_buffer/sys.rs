// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub use super::commands::{
    bind_push::UnsafeCommandBufferBuilderBindVertexBuffer,
    secondary::UnsafeCommandBufferBuilderExecuteCommands,
};
use super::{
    pool::CommandPoolAlloc, CommandBufferInheritanceInfo, CommandBufferLevel, CommandBufferUsage,
};
use crate::{
    command_buffer::{
        CommandBufferInheritanceRenderPassInfo, CommandBufferInheritanceRenderPassType,
        CommandBufferInheritanceRenderingInfo,
    },
    device::{Device, DeviceOwned},
    OomError, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
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
    #[inline]
    pub unsafe fn new(
        pool_alloc: &CommandPoolAlloc,
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
            let mut inheritance_info_vk = None;
            let mut inheritance_rendering_info_vk = None;
            let mut color_attachment_formats_vk: SmallVec<[_; 4]> = SmallVec::new();

            if let Some(inheritance_info) = &inheritance_info {
                let &CommandBufferInheritanceInfo {
                    ref render_pass,
                    occlusion_query,
                    query_statistics_flags,
                    _ne: _,
                } = inheritance_info;

                let inheritance_info_vk =
                    inheritance_info_vk.insert(ash::vk::CommandBufferInheritanceInfo {
                        render_pass: ash::vk::RenderPass::null(),
                        subpass: 0,
                        framebuffer: ash::vk::Framebuffer::null(),
                        occlusion_query_enable: ash::vk::FALSE,
                        query_flags: ash::vk::QueryControlFlags::empty(),
                        pipeline_statistics: query_statistics_flags.into(),
                        ..Default::default()
                    });

                if let Some(flags) = occlusion_query {
                    inheritance_info_vk.occlusion_query_enable = ash::vk::TRUE;

                    if flags.precise {
                        inheritance_info_vk.query_flags = ash::vk::QueryControlFlags::PRECISE;
                    }
                }

                if let Some(render_pass) = render_pass {
                    flags |= ash::vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE;

                    match render_pass {
                        CommandBufferInheritanceRenderPassType::BeginRenderPass(
                            render_pass_info,
                        ) => {
                            let &CommandBufferInheritanceRenderPassInfo {
                                ref subpass,
                                ref framebuffer,
                            } = render_pass_info;

                            inheritance_info_vk.render_pass = subpass.render_pass().handle();
                            inheritance_info_vk.subpass = subpass.index();
                            inheritance_info_vk.framebuffer = framebuffer
                                .as_ref()
                                .map(|fb| fb.handle())
                                .unwrap_or_default();
                        }
                        CommandBufferInheritanceRenderPassType::BeginRendering(rendering_info) => {
                            let &CommandBufferInheritanceRenderingInfo {
                                view_mask,
                                ref color_attachment_formats,
                                depth_attachment_format,
                                stencil_attachment_format,
                                rasterization_samples,
                            } = rendering_info;

                            color_attachment_formats_vk.extend(
                                color_attachment_formats.iter().map(|format| {
                                    format.map_or(ash::vk::Format::UNDEFINED, Into::into)
                                }),
                            );

                            let inheritance_rendering_info_vk = inheritance_rendering_info_vk
                                .insert(ash::vk::CommandBufferInheritanceRenderingInfo {
                                    flags: ash::vk::RenderingFlags::empty(),
                                    view_mask,
                                    color_attachment_count: color_attachment_formats_vk.len()
                                        as u32,
                                    p_color_attachment_formats: color_attachment_formats_vk
                                        .as_ptr(),
                                    depth_attachment_format: depth_attachment_format
                                        .map_or(ash::vk::Format::UNDEFINED, Into::into),
                                    stencil_attachment_format: stencil_attachment_format
                                        .map_or(ash::vk::Format::UNDEFINED, Into::into),
                                    rasterization_samples: rasterization_samples.into(),
                                    ..Default::default()
                                });

                            inheritance_info_vk.p_next =
                                inheritance_rendering_info_vk as *const _ as *const _;
                        }
                    }
                }
            }

            let begin_info_vk = ash::vk::CommandBufferBeginInfo {
                flags,
                p_inheritance_info: inheritance_info_vk
                    .as_ref()
                    .map_or(ptr::null(), |info| info),
                ..Default::default()
            };

            let fns = device.fns();

            (fns.v1_0.begin_command_buffer)(pool_alloc.handle(), &begin_info_vk)
                .result()
                .map_err(VulkanError::from)?;
        }

        Ok(UnsafeCommandBufferBuilder {
            handle: pool_alloc.handle(),
            device,
            usage,
        })
    }

    /// Turns the builder into an actual command buffer.
    #[inline]
    pub fn build(self) -> Result<UnsafeCommandBuffer, OomError> {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.end_command_buffer)(self.handle)
                .result()
                .map_err(VulkanError::from)?;

            Ok(UnsafeCommandBuffer {
                command_buffer: self.handle,
                device: self.device.clone(),
                usage: self.usage,
            })
        }
    }
}

unsafe impl VulkanObject for UnsafeCommandBufferBuilder {
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
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
#[derive(Debug)]
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
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.command_buffer
    }
}
