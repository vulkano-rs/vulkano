use super::{
    allocator::{
        CommandBufferAlloc, CommandBufferAllocator, CommandBufferBuilderAlloc,
        StandardCommandBufferAllocator,
    },
    CommandBufferInheritanceInfo, CommandBufferLevel, CommandBufferUsage,
};
use crate::{
    command_buffer::{
        CommandBufferInheritanceRenderPassInfo, CommandBufferInheritanceRenderPassType,
        CommandBufferInheritanceRenderingInfo,
    },
    device::{Device, DeviceOwned, QueueFamilyProperties},
    query::QueryControlFlags,
    ValidationError, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{fmt::Debug, ptr, sync::Arc};

/// Command buffer being built.
///
/// # Safety
///
/// - All submitted commands must be valid and follow the requirements of the Vulkan specification.
/// - Any resources used by submitted commands must outlive the returned builder and its created
///   command buffer. They must be protected against data races through manual synchronization.
pub struct UnsafeCommandBufferBuilder<A = StandardCommandBufferAllocator>
where
    A: CommandBufferAllocator,
{
    builder_alloc: A::Builder,

    queue_family_index: u32,
    // Must be `None` in a primary command buffer and `Some` in a secondary command buffer.
    inheritance_info: Option<CommandBufferInheritanceInfo>,
    pub(super) usage: CommandBufferUsage,
}

impl<A> UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    /// Creates a new builder, for recording commands.
    ///
    /// # Safety
    ///
    /// - `begin_info` must be valid.
    #[inline]
    pub unsafe fn new(
        allocator: &A,
        queue_family_index: u32,
        level: CommandBufferLevel,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<Self, VulkanError> {
        let builder_alloc = allocator
            .allocate(queue_family_index, level, 1)?
            .next()
            .expect("requested one command buffer from the command pool, but got zero");

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

            let fns = builder_alloc.device().fns();
            (fns.v1_0.begin_command_buffer)(builder_alloc.inner().handle(), &begin_info_vk)
                .result()
                .map_err(VulkanError::from)?;
        }

        Ok(UnsafeCommandBufferBuilder {
            builder_alloc,
            inheritance_info,
            queue_family_index,
            usage,
        })
    }

    /// Turns the builder into an actual command buffer.
    #[inline]
    pub fn build(self) -> Result<UnsafeCommandBuffer<A>, VulkanError> {
        unsafe {
            let fns = self.device().fns();
            (fns.v1_0.end_command_buffer)(self.handle())
                .result()
                .map_err(VulkanError::from)?;

            Ok(UnsafeCommandBuffer {
                alloc: self.builder_alloc.into_alloc(),
                inheritance_info: self.inheritance_info,
                queue_family_index: self.queue_family_index,
                usage: self.usage,
            })
        }
    }

    /// Returns the queue family index that this command buffer was created for.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Returns the level of the command buffer.
    #[inline]
    pub fn level(&self) -> CommandBufferLevel {
        self.builder_alloc.inner().level()
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

unsafe impl<A> VulkanObject for UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.builder_alloc.inner().handle()
    }
}

unsafe impl<A> DeviceOwned for UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.builder_alloc.device()
    }
}

impl<A> Debug for UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeCommandBufferBuilder")
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

/// Command buffer that has been built.
///
/// # Safety
///
/// The command buffer must not outlive the command pool that it was created from,
/// nor the resources used by the recorded commands.
#[derive(Debug)]
pub struct UnsafeCommandBuffer<A = StandardCommandBufferAllocator>
where
    A: CommandBufferAllocator,
{
    alloc: A::Alloc,

    queue_family_index: u32,
    // Must be `None` in a primary command buffer and `Some` in a secondary command buffer.
    inheritance_info: Option<CommandBufferInheritanceInfo>,
    usage: CommandBufferUsage,
}

impl<A> UnsafeCommandBuffer<A>
where
    A: CommandBufferAllocator,
{
    /// Returns the queue family index that this command buffer was created for.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Returns the level of the command buffer.
    #[inline]
    pub fn level(&self) -> CommandBufferLevel {
        self.alloc.inner().level()
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
}

unsafe impl<A> VulkanObject for UnsafeCommandBuffer<A>
where
    A: CommandBufferAllocator,
{
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.alloc.inner().handle()
    }
}

unsafe impl<A> DeviceOwned for UnsafeCommandBuffer<A>
where
    A: CommandBufferAllocator,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.alloc.device()
    }
}
