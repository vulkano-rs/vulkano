// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{CommandBufferBuilder, ExecuteCommandsError, RenderPassStateType};
use crate::{
    command_buffer::{
        allocator::{CommandBufferAlloc, CommandBufferAllocator},
        CommandBufferInheritanceRenderPassType, PrimaryCommandBuffer, SecondaryCommandBuffer,
        SubpassContents,
    },
    device::{DeviceOwned, QueueFlags},
    query::QueryType,
    RequiresOneOf, SafeDeref, VulkanObject,
};

impl<A> CommandBufferBuilder<PrimaryCommandBuffer<A::Alloc>, A>
where
    A: CommandBufferAllocator,
{
    /// Executes a secondary command buffer.
    ///
    /// If the `usage` that `command_buffer` was created with are more restrictive than those of
    /// `self`, then `self` will be restricted to match. E.g. executing a secondary command buffer
    /// with [`OneTimeSubmit`] will set `self`'s usage to
    /// `OneTimeSubmit` also.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers and images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    ///
    /// [`OneTimeSubmit`]: crate::command_buffer::CommandBufferUsage::OneTimeSubmit
    pub unsafe fn execute_commands(
        &mut self,
        command_buffer: SecondaryCommandBuffer<impl CommandBufferAlloc + 'static>,
    ) -> Result<&mut Self, ExecuteCommandsError> {
        self.validate_execute_commands(&command_buffer, 0)?;

        unsafe { Ok(self.execute_commands_unchecked(command_buffer)) }
    }

    fn validate_execute_commands(
        &self,
        command_buffer: &SecondaryCommandBuffer<impl CommandBufferAlloc + 'static>,
        command_buffer_index: u32,
    ) -> Result<(), ExecuteCommandsError> {
        // VUID-vkCmdExecuteCommands-commonparent
        assert_eq!(self.device(), command_buffer.device());

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdExecuteCommands-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(ExecuteCommandsError::NotSupportedByQueueFamily);
        }

        // TODO:
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00094

        if let Some(render_pass_state) = &self.builder_state.render_pass {
            // VUID-vkCmdExecuteCommands-contents-06018
            // VUID-vkCmdExecuteCommands-flags-06024
            if render_pass_state.contents != SubpassContents::SecondaryCommandBuffers {
                return Err(ExecuteCommandsError::ForbiddenWithSubpassContents {
                    contents: render_pass_state.contents,
                });
            }

            // VUID-vkCmdExecuteCommands-pCommandBuffers-00096
            let inheritance_render_pass = command_buffer
                .inheritance_info()
                .render_pass
                .as_ref()
                .ok_or(ExecuteCommandsError::RenderPassInheritanceRequired {
                    command_buffer_index,
                })?;

            match (&render_pass_state.render_pass, inheritance_render_pass) {
                (
                    RenderPassStateType::BeginRenderPass(state),
                    CommandBufferInheritanceRenderPassType::BeginRenderPass(inheritance_info),
                ) => {
                    // VUID-vkCmdExecuteCommands-pBeginInfo-06020
                    if !inheritance_info
                        .subpass
                        .render_pass()
                        .is_compatible_with(state.subpass.render_pass())
                    {
                        return Err(ExecuteCommandsError::RenderPassNotCompatible {
                            command_buffer_index,
                        });
                    }

                    // VUID-vkCmdExecuteCommands-pCommandBuffers-06019
                    if inheritance_info.subpass.index() != state.subpass.index() {
                        return Err(ExecuteCommandsError::RenderPassSubpassMismatch {
                            command_buffer_index,
                            required_subpass: state.subpass.index(),
                            inherited_subpass: inheritance_info.subpass.index(),
                        });
                    }

                    // VUID-vkCmdExecuteCommands-pCommandBuffers-00099
                    if let Some(framebuffer) = &inheritance_info.framebuffer {
                        if framebuffer != state.framebuffer.as_ref().unwrap() {
                            return Err(ExecuteCommandsError::RenderPassFramebufferMismatch {
                                command_buffer_index,
                            });
                        }
                    }
                }
                (
                    RenderPassStateType::BeginRendering(_),
                    CommandBufferInheritanceRenderPassType::BeginRendering(inheritance_info),
                ) => {
                    let attachments = render_pass_state.attachments.as_ref().unwrap();

                    // VUID-vkCmdExecuteCommands-colorAttachmentCount-06027
                    if inheritance_info.color_attachment_formats.len()
                        != attachments.color_attachments.len()
                    {
                        return Err(
                            ExecuteCommandsError::RenderPassColorAttachmentCountMismatch {
                                command_buffer_index,
                                required_count: attachments.color_attachments.len() as u32,
                                inherited_count: inheritance_info.color_attachment_formats.len()
                                    as u32,
                            },
                        );
                    }

                    for (color_attachment_index, image_view, inherited_format) in attachments
                        .color_attachments
                        .iter()
                        .zip(inheritance_info.color_attachment_formats.iter().copied())
                        .enumerate()
                        .filter_map(|(i, (a, f))| a.as_ref().map(|a| (i as u32, &a.image_view, f)))
                    {
                        let required_format = image_view.format().unwrap();

                        // VUID-vkCmdExecuteCommands-imageView-06028
                        if Some(required_format) != inherited_format {
                            return Err(
                                ExecuteCommandsError::RenderPassColorAttachmentFormatMismatch {
                                    command_buffer_index,
                                    color_attachment_index,
                                    required_format,
                                    inherited_format,
                                },
                            );
                        }

                        // VUID-vkCmdExecuteCommands-pNext-06035
                        if image_view.image().samples() != inheritance_info.rasterization_samples {
                            return Err(
                                ExecuteCommandsError::RenderPassColorAttachmentSamplesMismatch {
                                    command_buffer_index,
                                    color_attachment_index,
                                    required_samples: image_view.image().samples(),
                                    inherited_samples: inheritance_info.rasterization_samples,
                                },
                            );
                        }
                    }

                    if let Some((image_view, format)) = attachments
                        .depth_attachment
                        .as_ref()
                        .map(|a| (&a.image_view, inheritance_info.depth_attachment_format))
                    {
                        // VUID-vkCmdExecuteCommands-pDepthAttachment-06029
                        if Some(image_view.format().unwrap()) != format {
                            return Err(
                                ExecuteCommandsError::RenderPassDepthAttachmentFormatMismatch {
                                    command_buffer_index,
                                    required_format: image_view.format().unwrap(),
                                    inherited_format: format,
                                },
                            );
                        }

                        // VUID-vkCmdExecuteCommands-pNext-06036
                        if image_view.image().samples() != inheritance_info.rasterization_samples {
                            return Err(
                                ExecuteCommandsError::RenderPassDepthAttachmentSamplesMismatch {
                                    command_buffer_index,
                                    required_samples: image_view.image().samples(),
                                    inherited_samples: inheritance_info.rasterization_samples,
                                },
                            );
                        }
                    }

                    if let Some((image_view, format)) = attachments
                        .stencil_attachment
                        .as_ref()
                        .map(|a| (&a.image_view, inheritance_info.stencil_attachment_format))
                    {
                        // VUID-vkCmdExecuteCommands-pStencilAttachment-06030
                        if Some(image_view.format().unwrap()) != format {
                            return Err(
                                ExecuteCommandsError::RenderPassStencilAttachmentFormatMismatch {
                                    command_buffer_index,
                                    required_format: image_view.format().unwrap(),
                                    inherited_format: format,
                                },
                            );
                        }

                        // VUID-vkCmdExecuteCommands-pNext-06037
                        if image_view.image().samples() != inheritance_info.rasterization_samples {
                            return Err(
                                ExecuteCommandsError::RenderPassStencilAttachmentSamplesMismatch {
                                    command_buffer_index,
                                    required_samples: image_view.image().samples(),
                                    inherited_samples: inheritance_info.rasterization_samples,
                                },
                            );
                        }
                    }

                    // VUID-vkCmdExecuteCommands-viewMask-06031
                    if inheritance_info.view_mask != render_pass_state.rendering_info.view_mask {
                        return Err(ExecuteCommandsError::RenderPassViewMaskMismatch {
                            command_buffer_index,
                            required_view_mask: render_pass_state.rendering_info.view_mask,
                            inherited_view_mask: inheritance_info.view_mask,
                        });
                    }
                }
                _ => {
                    // VUID-vkCmdExecuteCommands-pBeginInfo-06025
                    return Err(ExecuteCommandsError::RenderPassTypeMismatch {
                        command_buffer_index,
                    });
                }
            }

            // TODO:
            // VUID-vkCmdExecuteCommands-commandBuffer-06533
            // VUID-vkCmdExecuteCommands-commandBuffer-06534
            // VUID-vkCmdExecuteCommands-pCommandBuffers-06535
            // VUID-vkCmdExecuteCommands-pCommandBuffers-06536
        } else {
            // VUID-vkCmdExecuteCommands-pCommandBuffers-00100
            if command_buffer.inheritance_info().render_pass.is_some() {
                return Err(ExecuteCommandsError::RenderPassInheritanceForbidden {
                    command_buffer_index,
                });
            }
        }

        // VUID-vkCmdExecuteCommands-commandBuffer-00101
        if !self.builder_state.queries.is_empty()
            && !self.device().enabled_features().inherited_queries
        {
            return Err(ExecuteCommandsError::RequirementNotMet {
                required_for: "`CommandBufferBuilder::execute_commands` when a query is active",
                requires_one_of: RequiresOneOf {
                    features: &["inherited_queries"],
                    ..Default::default()
                },
            });
        }

        for state in self.builder_state.queries.values() {
            match state.ty {
                QueryType::Occlusion => {
                    // VUID-vkCmdExecuteCommands-commandBuffer-00102
                    let inherited_flags = command_buffer.inheritance_info().occlusion_query.ok_or(
                        ExecuteCommandsError::OcclusionQueryInheritanceRequired {
                            command_buffer_index,
                        },
                    )?;

                    let inherited_flags_vk = ash::vk::QueryControlFlags::from(inherited_flags);
                    let state_flags_vk = ash::vk::QueryControlFlags::from(state.flags);

                    // VUID-vkCmdExecuteCommands-commandBuffer-00103
                    if inherited_flags_vk & state_flags_vk != state_flags_vk {
                        return Err(ExecuteCommandsError::OcclusionQueryFlagsNotSuperset {
                            command_buffer_index,
                            required_flags: state.flags,
                            inherited_flags,
                        });
                    }
                }
                QueryType::PipelineStatistics(state_flags) => {
                    let inherited_flags = command_buffer.inheritance_info().query_statistics_flags;
                    let inherited_flags_vk =
                        ash::vk::QueryPipelineStatisticFlags::from(inherited_flags);
                    let state_flags_vk = ash::vk::QueryPipelineStatisticFlags::from(state_flags);

                    // VUID-vkCmdExecuteCommands-commandBuffer-00104
                    if inherited_flags_vk & state_flags_vk != state_flags_vk {
                        return Err(
                            ExecuteCommandsError::PipelineStatisticsQueryFlagsNotSuperset {
                                command_buffer_index,
                                required_flags: state_flags,
                                inherited_flags,
                            },
                        );
                    }
                }
                QueryType::Timestamp => (),
            }
        }

        // TODO:
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00091
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00092
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00093
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00105

        // VUID-vkCmdExecuteCommands-bufferlevel
        // Ensured by the type of the impl block.

        // VUID-vkCmdExecuteCommands-pCommandBuffers-00088
        // VUID-vkCmdExecuteCommands-pCommandBuffers-00089
        // Ensured by the SecondaryCommandBuffer trait.

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn execute_commands_unchecked(
        &mut self,
        command_buffer: SecondaryCommandBuffer<impl CommandBufferAlloc + 'static>,
    ) -> &mut Self {
        struct DropUnlock<As>(SecondaryCommandBuffer<As>)
        where
            As: CommandBufferAlloc;

        impl<As> std::ops::Deref for DropUnlock<As>
        where
            As: CommandBufferAlloc,
        {
            type Target = SecondaryCommandBuffer<As>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        unsafe impl<As> SafeDeref for DropUnlock<As> where As: CommandBufferAlloc {}

        impl<As> Drop for DropUnlock<As>
        where
            As: CommandBufferAlloc,
        {
            fn drop(&mut self) {
                unsafe {
                    self.unlock();
                }
            }
        }

        let command_buffer = {
            command_buffer.lock_record().unwrap();
            DropUnlock(command_buffer)
        };

        let fns = self.device().fns();
        (fns.v1_0.cmd_execute_commands)(self.handle(), 1, &command_buffer.handle());

        // The secondary command buffer could leave the primary in any state.
        self.builder_state = Default::default();

        // If the secondary is non-concurrent or one-time use, that restricts the primary as well.
        self.usage = std::cmp::min(self.usage, command_buffer.usage);

        let _command_index = self.next_command_index;
        let _command_name = "execute_commands";

        // TODO: sync state update

        self.resources.push(Box::new(command_buffer));

        self.next_command_index += 1;
        self
    }
}
