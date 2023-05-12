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
        allocator::CommandBufferAllocator,
        auto::{RenderPassStateType, Resource, ResourceUseRef2},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, CommandBufferExecError, CommandBufferInheritanceRenderPassType,
        ResourceInCommand, SecondaryCommandBufferAbstract, SecondaryCommandBufferBufferUsage,
        SecondaryCommandBufferImageUsage, SecondaryCommandBufferResourcesUsage, SubpassContents,
    },
    device::{DeviceOwned, QueueFlags},
    format::Format,
    image::SampleCount,
    query::{QueryControlFlags, QueryPipelineStatisticFlags, QueryType},
    RequiresOneOf, SafeDeref, VulkanObject,
};
use smallvec::{smallvec, SmallVec};
use std::{
    cmp::min,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    sync::Arc,
};

/// # Commands to execute a secondary command buffer inside a primary command buffer.
///
/// These commands can be called on any queue that can execute the commands recorded in the
/// secondary command buffer.
impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Executes a secondary command buffer.
    ///
    /// If the `flags` that `command_buffer` was created with are more restrictive than those of
    /// `self`, then `self` will be restricted to match. E.g. executing a secondary command buffer
    /// with `Flags::OneTimeSubmit` will set `self`'s flags to `Flags::OneTimeSubmit` also.
    pub fn execute_commands(
        &mut self,
        command_buffer: Arc<dyn SecondaryCommandBufferAbstract>,
    ) -> Result<&mut Self, ExecuteCommandsError> {
        let command_buffer = DropUnlockCommandBuffer::new(command_buffer)?;
        self.validate_execute_commands(&command_buffer, 0)?;

        unsafe {
            self.execute_commands_locked(smallvec![command_buffer]);
        }

        Ok(self)
    }

    /// Executes multiple secondary command buffers in a vector.
    ///
    /// This requires that the secondary command buffers do not have resource conflicts; an error
    /// will be returned if there are any. Use `execute_commands` if you want to ensure that
    /// resource conflicts are automatically resolved.
    // TODO ^ would be nice if this just worked without errors
    pub fn execute_commands_from_vec(
        &mut self,
        command_buffers: Vec<Arc<dyn SecondaryCommandBufferAbstract>>,
    ) -> Result<&mut Self, ExecuteCommandsError> {
        let command_buffers: SmallVec<[_; 4]> = command_buffers
            .into_iter()
            .map(DropUnlockCommandBuffer::new)
            .collect::<Result<_, _>>()?;

        for (command_buffer_index, command_buffer) in command_buffers.iter().enumerate() {
            self.validate_execute_commands(command_buffer, command_buffer_index as u32)?;
        }

        unsafe {
            self.execute_commands_locked(command_buffers);
        }

        Ok(self)
    }

    fn validate_execute_commands(
        &self,
        command_buffer: &dyn SecondaryCommandBufferAbstract,
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
                required_for: "`AutoCommandBufferBuilder::execute_commands` when a query is active",
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

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn execute_commands_unchecked(
        &mut self,
        command_buffers: impl IntoIterator<Item = Arc<dyn SecondaryCommandBufferAbstract>>,
    ) -> &mut Self {
        self.execute_commands_locked(
            command_buffers
                .into_iter()
                .map(DropUnlockCommandBuffer::new)
                .collect::<Result<_, _>>()
                .unwrap(),
        )
    }

    unsafe fn execute_commands_locked(
        &mut self,
        command_buffers: SmallVec<[DropUnlockCommandBuffer; 4]>,
    ) -> &mut Self {
        // Secondary command buffer could leave the primary in any state.
        self.builder_state.reset_non_render_pass_states();

        self.add_command(
            "execute_commands",
            command_buffers
                .iter()
                .enumerate()
                .flat_map(|(index, command_buffer)| {
                    let index = index as u32;
                    let SecondaryCommandBufferResourcesUsage { buffers, images } =
                        command_buffer.resources_usage();

                    (buffers.iter().map(move |usage| {
                        let &SecondaryCommandBufferBufferUsage {
                            use_ref,
                            ref buffer,
                            ref range,
                            memory,
                        } = usage;

                        (
                            ResourceUseRef2 {
                                resource_in_command: ResourceInCommand::SecondaryCommandBuffer {
                                    index,
                                },
                                secondary_use_ref: Some(use_ref.into()),
                            },
                            Resource::Buffer {
                                buffer: buffer.clone(),
                                range: range.clone(),
                                memory,
                            },
                        )
                    }))
                    .chain(images.iter().map(move |usage| {
                        let &SecondaryCommandBufferImageUsage {
                            use_ref,
                            ref image,
                            ref subresource_range,
                            memory,
                            start_layout,
                            end_layout,
                        } = usage;

                        (
                            ResourceUseRef2 {
                                resource_in_command: ResourceInCommand::SecondaryCommandBuffer {
                                    index,
                                },
                                secondary_use_ref: Some(use_ref.into()),
                            },
                            Resource::Image {
                                image: image.clone(),
                                subresource_range: subresource_range.clone(),
                                memory,
                                start_layout,
                                end_layout,
                            },
                        )
                    }))
                })
                .collect(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.execute_commands_locked(command_buffers);
            },
        );

        self
    }
}

impl<A> UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    /// Calls `vkCmdExecuteCommands` on the builder.
    ///
    /// Does nothing if the list of command buffers is empty, as it would be a no-op and isn't a
    /// valid usage of the command anyway.
    #[inline]
    pub unsafe fn execute_commands(
        &mut self,
        command_buffers: impl IntoIterator<Item = Arc<dyn SecondaryCommandBufferAbstract>>,
    ) -> &mut Self {
        self.execute_commands_locked(
            command_buffers
                .into_iter()
                .map(DropUnlockCommandBuffer::new)
                .collect::<Result<_, _>>()
                .unwrap(),
        );

        self
    }

    unsafe fn execute_commands_locked(
        &mut self,
        command_buffers: SmallVec<[DropUnlockCommandBuffer; 4]>,
    ) -> &mut Self {
        if command_buffers.is_empty() {
            return self;
        }

        let command_buffers_vk: SmallVec<[_; 4]> =
            command_buffers.iter().map(|cb| cb.handle()).collect();

        let fns = self.device().fns();
        (fns.v1_0.cmd_execute_commands)(
            self.handle(),
            command_buffers_vk.len() as u32,
            command_buffers_vk.as_ptr(),
        );

        // If the secondary is non-concurrent or one-time use, that restricts the primary as
        // well.
        self.usage = command_buffers
            .iter()
            .map(|cb| cb.usage())
            .fold(self.usage, min);

        self.keep_alive_objects
            .extend(command_buffers.into_iter().map(|cb| Box::new(cb) as _));

        self
    }
}

struct DropUnlockCommandBuffer(Arc<dyn SecondaryCommandBufferAbstract>);

impl DropUnlockCommandBuffer {
    fn new(
        command_buffer: Arc<dyn SecondaryCommandBufferAbstract>,
    ) -> Result<Self, CommandBufferExecError> {
        command_buffer.lock_record()?;
        Ok(Self(command_buffer))
    }
}

impl std::ops::Deref for DropUnlockCommandBuffer {
    type Target = Arc<dyn SecondaryCommandBufferAbstract>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
unsafe impl SafeDeref for DropUnlockCommandBuffer {}

impl Drop for DropUnlockCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.unlock();
        }
    }
}

/// Error that can happen when executing a secondary command buffer.
#[derive(Clone, Debug)]
pub enum ExecuteCommandsError {
    ExecError(CommandBufferExecError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// Operation forbidden inside a render subpass with the specified contents.
    ForbiddenWithSubpassContents {
        contents: SubpassContents,
    },

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,

    /// A render pass is active, but a command buffer does not contain occlusion query inheritance
    /// info.
    OcclusionQueryInheritanceRequired {
        command_buffer_index: u32,
    },

    /// The inherited occlusion query control flags of a command buffer are not a superset of the
    /// currently active flags.
    OcclusionQueryFlagsNotSuperset {
        command_buffer_index: u32,
        required_flags: QueryControlFlags,
        inherited_flags: QueryControlFlags,
    },

    /// The inherited pipeline statistics query flags of a command buffer are not a superset of the
    /// currently active flags.
    PipelineStatisticsQueryFlagsNotSuperset {
        command_buffer_index: u32,
        required_flags: QueryPipelineStatisticFlags,
        inherited_flags: QueryPipelineStatisticFlags,
    },

    /// The inherited color attachment count of a command buffer does not match the current
    /// attachment count.
    RenderPassColorAttachmentCountMismatch {
        command_buffer_index: u32,
        required_count: u32,
        inherited_count: u32,
    },

    /// The inherited format of a color attachment of a command buffer does not match the current
    /// attachment format.
    RenderPassColorAttachmentFormatMismatch {
        command_buffer_index: u32,
        color_attachment_index: u32,
        required_format: Format,
        inherited_format: Option<Format>,
    },

    /// The inherited sample count of a color attachment of a command buffer does not match the
    /// current attachment sample count.
    RenderPassColorAttachmentSamplesMismatch {
        command_buffer_index: u32,
        color_attachment_index: u32,
        required_samples: SampleCount,
        inherited_samples: SampleCount,
    },

    /// The inherited format of the depth attachment of a command buffer does not match the current
    /// attachment format.
    RenderPassDepthAttachmentFormatMismatch {
        command_buffer_index: u32,
        required_format: Format,
        inherited_format: Option<Format>,
    },

    /// The inherited sample count of the depth attachment of a command buffer does not match the
    /// current attachment sample count.
    RenderPassDepthAttachmentSamplesMismatch {
        command_buffer_index: u32,
        required_samples: SampleCount,
        inherited_samples: SampleCount,
    },

    /// The inherited framebuffer of a command buffer does not match the current framebuffer.
    RenderPassFramebufferMismatch {
        command_buffer_index: u32,
    },

    /// A render pass is active, but a command buffer does not contain render pass inheritance info.
    RenderPassInheritanceRequired {
        command_buffer_index: u32,
    },

    /// A render pass is not active, but a command buffer contains render pass inheritance info.
    RenderPassInheritanceForbidden {
        command_buffer_index: u32,
    },

    /// The inherited render pass of a command buffer is not compatible with the current render
    /// pass.
    RenderPassNotCompatible {
        command_buffer_index: u32,
    },

    /// The inherited format of the stencil attachment of a command buffer does not match the
    /// current attachment format.
    RenderPassStencilAttachmentFormatMismatch {
        command_buffer_index: u32,
        required_format: Format,
        inherited_format: Option<Format>,
    },

    /// The inherited sample count of the stencil attachment of a command buffer does not match the
    /// current attachment sample count.
    RenderPassStencilAttachmentSamplesMismatch {
        command_buffer_index: u32,
        required_samples: SampleCount,
        inherited_samples: SampleCount,
    },

    /// The inherited subpass index of a command buffer does not match the current subpass index.
    RenderPassSubpassMismatch {
        command_buffer_index: u32,
        required_subpass: u32,
        inherited_subpass: u32,
    },

    /// The inherited render pass of a command buffer is of the wrong type.
    RenderPassTypeMismatch {
        command_buffer_index: u32,
    },

    /// The inherited view mask of a command buffer does not match the current view mask.
    RenderPassViewMaskMismatch {
        command_buffer_index: u32,
        required_view_mask: u32,
        inherited_view_mask: u32,
    },
}

impl Error for ExecuteCommandsError {}

impl Display for ExecuteCommandsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::ExecError(err) => Display::fmt(err, f),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::ForbiddenWithSubpassContents {
                contents: subpass_contents,
            } => write!(
                f,
                "operation forbidden inside a render subpass with contents {:?}",
                subpass_contents,
            ),
            Self::NotSupportedByQueueFamily => {
                write!(f, "the queue family doesn't allow this operation")
            }
            Self::OcclusionQueryInheritanceRequired {
                command_buffer_index,
            } => write!(
                f,
                "a render pass is active, but command buffer {} does not contain occlusion query \
                inheritance info",
                command_buffer_index,
            ),
            Self::OcclusionQueryFlagsNotSuperset {
                command_buffer_index,
                required_flags,
                inherited_flags,
            } => write!(
                f,
                "the inherited occlusion query control flags ({:?}) of command buffer {} are not a \
                superset of the currently active flags ({:?})",
                inherited_flags, command_buffer_index, required_flags,
            ),
            Self::PipelineStatisticsQueryFlagsNotSuperset {
                command_buffer_index,
                required_flags,
                inherited_flags,
            } => write!(
                f,
                "the inherited pipeline statistics query flags ({:?}) of command buffer {} are not \
                a superset of the currently active flags ({:?})",
                inherited_flags, command_buffer_index, required_flags,
            ),
            Self::RenderPassColorAttachmentCountMismatch {
                command_buffer_index,
                required_count,
                inherited_count,
            } => write!(
                f,
                "the inherited color attachment count ({}) of command buffer {} does not match the \
                current attachment count ({})",
                inherited_count, command_buffer_index, required_count,
            ),
            Self::RenderPassColorAttachmentFormatMismatch {
                command_buffer_index,
                color_attachment_index,
                required_format,
                inherited_format,
            } => write!(
                f,
                "the inherited format ({:?}) of color attachment {} of command buffer {} does not \
                match the current attachment format ({:?})",
                inherited_format, color_attachment_index, command_buffer_index, required_format,
            ),
            Self::RenderPassColorAttachmentSamplesMismatch {
                command_buffer_index,
                color_attachment_index,
                required_samples,
                inherited_samples,
            } => write!(
                f,
                "the inherited sample count ({:?}) of color attachment {} of command buffer {} \
                does not match the current attachment sample count ({:?})",
                inherited_samples, color_attachment_index, command_buffer_index, required_samples,
            ),
            Self::RenderPassDepthAttachmentFormatMismatch {
                command_buffer_index,
                required_format,
                inherited_format,
            } => write!(
                f,
                "the inherited format ({:?}) of the depth attachment of command buffer {} does not \
                match the current attachment format ({:?})",
                inherited_format, command_buffer_index, required_format,
            ),
            Self::RenderPassDepthAttachmentSamplesMismatch {
                command_buffer_index,
                required_samples,
                inherited_samples,
            } => write!(
                f,
                "the inherited sample count ({:?}) of the depth attachment of command buffer {} \
                does not match the current attachment sample count ({:?})",
                inherited_samples, command_buffer_index, required_samples,
            ),
            Self::RenderPassFramebufferMismatch {
                command_buffer_index,
            } => write!(
                f,
                "the inherited framebuffer of command buffer {} does not match the current \
                framebuffer",
                command_buffer_index,
            ),
            Self::RenderPassInheritanceRequired {
                command_buffer_index,
            } => write!(
                f,
                "a render pass is active, but command buffer {} does not contain render pass \
                inheritance info",
                command_buffer_index,
            ),
            Self::RenderPassInheritanceForbidden {
                command_buffer_index,
            } => write!(
                f,
                "a render pass is not active, but command buffer {} contains render pass \
                inheritance info",
                command_buffer_index,
            ),
            Self::RenderPassNotCompatible {
                command_buffer_index,
            } => write!(
                f,
                "the inherited render pass of command buffer {} is not compatible with the current \
                render pass",
                command_buffer_index,
            ),
            Self::RenderPassStencilAttachmentFormatMismatch {
                command_buffer_index,
                required_format,
                inherited_format,
            } => write!(
                f,
                "the inherited format ({:?}) of the stencil attachment of command buffer {} does \
                not match the current attachment format ({:?})",
                inherited_format, command_buffer_index, required_format,
            ),
            Self::RenderPassStencilAttachmentSamplesMismatch {
                command_buffer_index,
                required_samples,
                inherited_samples,
            } => write!(
                f,
                "the inherited sample count ({:?}) of the stencil attachment of command buffer {} \
                does not match the current attachment sample count ({:?})",
                inherited_samples, command_buffer_index, required_samples,
            ),
            Self::RenderPassSubpassMismatch {
                command_buffer_index,
                required_subpass,
                inherited_subpass,
            } => write!(
                f,
                "the inherited subpass index ({}) of command buffer {} does not match the current \
                subpass index ({})",
                inherited_subpass, command_buffer_index, required_subpass,
            ),
            Self::RenderPassTypeMismatch {
                command_buffer_index,
            } => write!(
                f,
                "the inherited render pass of command buffer {} is of the wrong type",
                command_buffer_index,
            ),
            Self::RenderPassViewMaskMismatch {
                command_buffer_index,
                required_view_mask,
                inherited_view_mask,
            } => write!(
                f,
                "the inherited view mask ({}) of command buffer {} does not match the current view \
                mask ({})",
                inherited_view_mask, command_buffer_index, required_view_mask,
            ),
        }
    }
}

impl From<CommandBufferExecError> for ExecuteCommandsError {
    fn from(val: CommandBufferExecError) -> Self {
        Self::ExecError(val)
    }
}
