use super::{
    allocator::{CommandBufferAlloc, CommandBufferAllocator},
    CommandBufferInheritanceInfo, CommandBufferLevel, CommandBufferUsage,
};
use crate::{
    command_buffer::{
        CommandBufferInheritanceRenderPassInfo, CommandBufferInheritanceRenderPassType,
        CommandBufferInheritanceRenderingInfo,
    },
    device::{Device, DeviceOwned, QueueFamilyProperties},
    query::QueryControlFlags,
    Validated, ValidationError, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{fmt::Debug, mem::ManuallyDrop, ptr, sync::Arc};

/// A raw command buffer in the recording state.
///
/// This type corresponds directly to a `VkCommandBuffer` after it has been allocated and started
/// recording. It doesn't keep track of synchronization or resource lifetimes. As such, all
/// recorded commands are unsafe and it is the user's duty to make sure that data races are
/// protected against using manual synchronization and all resources used by the recorded commands
/// outlive the command buffer.
///
/// Note that command buffers in the recording state don't implement the `Send` and `Sync` traits.
/// Once a command buffer has finished recording, however, it *does* implement `Send` and `Sync`.
pub struct RawRecordingCommandBuffer {
    allocation: ManuallyDrop<CommandBufferAlloc>,
    allocator: Arc<dyn CommandBufferAllocator>,
    queue_family_index: u32,
    // Must be `None` in a primary command buffer and `Some` in a secondary command buffer.
    inheritance_info: Option<CommandBufferInheritanceInfo>,
    pub(super) usage: CommandBufferUsage,
}

impl RawRecordingCommandBuffer {
    /// Allocates and begins recording a new command buffer.
    #[inline]
    pub fn new(
        allocator: Arc<dyn CommandBufferAllocator>,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<Self, Validated<VulkanError>> {
        Self::validate_new(allocator.device(), queue_family_index, level, &begin_info)?;

        unsafe { Self::new_unchecked(allocator, queue_family_index, level, begin_info) }
    }

    pub(super) fn validate_new(
        device: &Device,
        _queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: &CommandBufferBeginInfo,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkBeginCommandBuffer-commandBuffer-00049
        // VUID-vkBeginCommandBuffer-commandBuffer-00050
        // Guaranteed by `CommandBufferAllocator`.

        if level == CommandBufferLevel::Secondary && begin_info.inheritance_info.is_none() {
            return Err(Box::new(ValidationError {
                context: "begin_info.inheritance_info".into(),
                problem: "is `None` while `level` is `CommandBufferLevel::Secondary`".into(),
                vuids: &["VUID-vkBeginCommandBuffer-commandBuffer-00051"],
                ..Default::default()
            }));
        }

        begin_info
            .validate(device)
            .map_err(|err| err.add_context("begin_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        allocator: Arc<dyn CommandBufferAllocator>,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<Self, Validated<VulkanError>> {
        let allocation = allocator.allocate(queue_family_index, level)?;

        let CommandBufferBeginInfo {
            usage,
            inheritance_info,
            _ne: _,
        } = begin_info;

        {
            let mut flags = ash::vk::CommandBufferUsageFlags::from(usage);
            let mut inheritance_info_vk = None;
            let mut inheritance_rendering_info_vk = None;
            let mut color_attachment_formats_vk: SmallVec<[_; 4]> = SmallVec::new();

            if let Some(inheritance_info) = &inheritance_info {
                let &CommandBufferInheritanceInfo {
                    ref render_pass,
                    occlusion_query,
                    pipeline_statistics,
                    _ne: _,
                } = inheritance_info;

                let inheritance_info_vk =
                    inheritance_info_vk.insert(ash::vk::CommandBufferInheritanceInfo {
                        render_pass: ash::vk::RenderPass::null(),
                        subpass: 0,
                        framebuffer: ash::vk::Framebuffer::null(),
                        occlusion_query_enable: ash::vk::FALSE,
                        query_flags: ash::vk::QueryControlFlags::empty(),
                        pipeline_statistics: pipeline_statistics.into(),
                        ..Default::default()
                    });

                if let Some(flags) = occlusion_query {
                    inheritance_info_vk.occlusion_query_enable = ash::vk::TRUE;

                    if flags.intersects(QueryControlFlags::PRECISE) {
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
                                ptr::from_ref(inheritance_rendering_info_vk).cast();
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

            let fns = allocation.inner.device().fns();
            (fns.v1_0.begin_command_buffer)(allocation.inner.handle(), &begin_info_vk)
                .result()
                .map_err(VulkanError::from)?;
        }

        Ok(RawRecordingCommandBuffer {
            allocation: ManuallyDrop::new(allocation),
            allocator,
            inheritance_info,
            queue_family_index,
            usage,
        })
    }

    /// Ends the recording, returning a command buffer which can be submitted.
    #[inline]
    pub unsafe fn end(self) -> Result<RawCommandBuffer, VulkanError> {
        let fns = self.device().fns();
        (fns.v1_0.end_command_buffer)(self.handle())
            .result()
            .map_err(VulkanError::from)?;

        Ok(RawCommandBuffer { inner: self })
    }

    /// Returns the queue family index that this command buffer was created for.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Returns the level of the command buffer.
    #[inline]
    pub fn level(&self) -> CommandBufferLevel {
        self.allocation.inner.level()
    }

    /// Returns the usage that the command buffer was created with.
    #[inline]
    pub fn usage(&self) -> CommandBufferUsage {
        self.usage
    }

    /// Returns the inheritance info of the command buffer, if it is a secondary command buffer.
    #[inline]
    pub fn inheritance_info(&self) -> Option<&CommandBufferInheritanceInfo> {
        self.inheritance_info.as_ref()
    }

    pub(in crate::command_buffer) fn queue_family_properties(&self) -> &QueueFamilyProperties {
        &self.device().physical_device().queue_family_properties()[self.queue_family_index as usize]
    }
}

impl Drop for RawRecordingCommandBuffer {
    #[inline]
    fn drop(&mut self) {
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        unsafe { self.allocator.deallocate(allocation) };
    }
}

unsafe impl VulkanObject for RawRecordingCommandBuffer {
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.allocation.inner.handle()
    }
}

unsafe impl DeviceOwned for RawRecordingCommandBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.allocation.inner.device()
    }
}

impl Debug for RawRecordingCommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawRecordingCommandBuffer")
            .field("handle", &self.level())
            .field("level", &self.level())
            .field("usage", &self.usage)
            .finish()
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

impl CommandBufferBeginInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            usage: _,
            ref inheritance_info,
            _ne: _,
        } = &self;

        if let Some(inheritance_info) = &inheritance_info {
            inheritance_info
                .validate(device)
                .map_err(|err| err.add_context("inheritance_info"))?;
        } else {
            // VUID-vkBeginCommandBuffer-commandBuffer-02840
            // Ensured by the definition of the `CommandBufferUsage` enum.
        }

        Ok(())
    }
}

/// A raw command buffer that has finished recording.
#[derive(Debug)]
pub struct RawCommandBuffer {
    inner: RawRecordingCommandBuffer,
}

// `RawRecordingCommandBuffer` is `!Send + !Sync` so that the implementation of
// `CommandBufferAllocator::allocate` can assume that a command buffer in the recording state
// doesn't leave the thread it was allocated on. However, as the safety contract states,
// `CommandBufferAllocator::deallocate` must acccount for the possibility that a command buffer is
// moved between threads after the recording is finished, and thus deallocated from another thread.
// That's why this is sound.
unsafe impl Send for RawCommandBuffer {}
unsafe impl Sync for RawCommandBuffer {}

impl RawCommandBuffer {
    /// Returns the queue family index that this command buffer was created for.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.inner.queue_family_index
    }

    /// Returns the level of the command buffer.
    #[inline]
    pub fn level(&self) -> CommandBufferLevel {
        self.inner.allocation.inner.level()
    }

    /// Returns the usage that the command buffer was created with.
    #[inline]
    pub fn usage(&self) -> CommandBufferUsage {
        self.inner.usage
    }

    /// Returns the inheritance info of the command buffer, if it is a secondary command buffer.
    #[inline]
    pub fn inheritance_info(&self) -> Option<&CommandBufferInheritanceInfo> {
        self.inner.inheritance_info.as_ref()
    }
}

unsafe impl VulkanObject for RawCommandBuffer {
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.allocation.inner.handle()
    }
}

unsafe impl DeviceOwned for RawCommandBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.allocation.inner.device()
    }
}
