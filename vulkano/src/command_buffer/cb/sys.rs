// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ptr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use buffer::BufferAccess;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferBuilder;
use command_buffer::CommandBufferExecError;
use command_buffer::cb::CommandBufferBuild;
use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolBuilderAlloc;
use command_buffer::pool::CommandPoolAlloc;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use framebuffer::EmptySinglePassRenderPassDesc;
use framebuffer::Framebuffer;
use framebuffer::FramebufferAbstract;
use framebuffer::RenderPass;
use framebuffer::RenderPassAbstract;
use framebuffer::Subpass;
use image::Layout;
use image::ImageAccess;
use instance::QueueFamily;
use sync::AccessCheckError;
use sync::AccessFlagBits;
use sync::PipelineStages;
use sync::GpuFuture;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Determines the kind of command buffer that we want to create.
#[derive(Debug, Clone)]
pub enum Kind<R, F> {
    /// A primary command buffer can execute all commands and can call secondary command buffers.
    Primary,

    /// A secondary command buffer can execute all dispatch and transfer operations, but not
    /// drawing operations.
    Secondary,

    /// A secondary command buffer within a render pass can only call draw operations that can
    /// be executed from within a specific subpass.
    SecondaryRenderPass {
        /// Which subpass this secondary command buffer can be called from.
        subpass: Subpass<R>,

        /// The framebuffer object that will be used when calling the command buffer.
        /// This parameter is optional and is an optimization hint for the implementation.
        framebuffer: Option<F>,
    },
}

impl Kind<RenderPass<EmptySinglePassRenderPassDesc>, Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>> {
    /// Equivalent to `Kind::Primary`.
    ///
    /// > **Note**: If you use `let kind = Kind::Primary;` in your code, you will probably get a
    /// > compilation error because the Rust compiler couldn't determine the template parameters
    /// > of `Kind`. To solve that problem in an easy way you can use this function instead.
    #[inline]
    pub fn primary() -> Kind<RenderPass<EmptySinglePassRenderPassDesc>, Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>> {
        Kind::Primary
    }
}

/// Flags to pass when creating a command buffer.
///
/// The safest option is `SimultaneousUse`, but it may be slower than the other two.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Flags {
    /// The command buffer can be used multiple times, but must not execute more than once
    /// simultaneously.
    None,

    /// The command buffer can be executed multiple times in parallel.
    SimultaneousUse,

    /// The command buffer can only be submitted once. Any further submit is forbidden.
    OneTimeSubmit,
}

/// Command buffer being built.
///
/// You can add commands to an `UnsafeCommandBufferBuilder` by using the `AddCommand` trait.
/// The `AddCommand<&Cmd>` trait is implemented on the `UnsafeCommandBufferBuilder` for any `Cmd`
/// that is a raw Vulkan command.
///
/// When you are finished adding commands, you can use the `CommandBufferBuild` trait to turn this
/// builder into an `UnsafeCommandBuffer`.
// TODO: change P parameter to be a CommandPoolBuilderAlloc
pub struct UnsafeCommandBufferBuilder<P> where P: CommandPool {
    // The command buffer obtained from the pool. Contains `None` if `build()` has been called.
    cmd: Option<P::Builder>,

    // Device that owns the command buffer.
    // TODO: necessary?
    device: Arc<Device>,

    // Flags that were used at creation.
    // TODO: necessary?
    flags: Flags,

    // True if we are a secondary command buffer.
    // TODO: necessary?
    secondary_cb: bool,
}

impl<P> UnsafeCommandBufferBuilder<P> where P: CommandPool {
    /// Creates a new builder.
    ///
    /// # Safety
    ///
    /// Creating and destroying an unsafe command buffer is not unsafe per se, but the commands
    /// that you add to it are unchecked, do not have any synchronization, and are not kept alive.
    ///
    /// In other words, it is your job to make sure that the commands you add are valid, that they
    /// don't use resources that have been destroyed, and that they do not introduce any race
    /// condition.
    ///
    /// > **Note**: Some checks are still made with `debug_assert!`. Do not expect to be able to
    /// > submit invalid commands.
    pub unsafe fn new<R, F>(pool: &P, kind: Kind<R, F>, flags: Flags)
                            -> Result<UnsafeCommandBufferBuilder<P>, OomError>
        where R: RenderPassAbstract, F: FramebufferAbstract
    {
        let secondary = match kind {
            Kind::Primary => false,
            Kind::Secondary | Kind::SecondaryRenderPass { .. } => true,
        };

        let cmd = try!(pool.alloc(secondary, 1)).next().expect("Requested one command buffer from \
                                                                the command pool, but got zero.");
        UnsafeCommandBufferBuilder::already_allocated(cmd, kind, flags)
    }

    /// Creates a new command buffer builder from an already-allocated command buffer.
    ///
    /// # Safety
    ///
    /// See the `new` method.
    ///
    /// The kind must match how the command buffer was allocated.
    ///
    pub unsafe fn already_allocated<R, F>(alloc: P::Builder, kind: Kind<R, F>, flags: Flags)
                                          -> Result<UnsafeCommandBufferBuilder<P>, OomError>
        where R: RenderPassAbstract, F: FramebufferAbstract
    {
        let device = alloc.device().clone();
        let vk = device.pointers();
        let cmd = alloc.inner().internal_object();

        let vk_flags = {
            let a = match flags {
                Flags::None => 0,
                Flags::SimultaneousUse => vk::COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
                Flags::OneTimeSubmit => vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            let b = match kind {
                Kind::Primary | Kind::Secondary => 0,
                Kind::SecondaryRenderPass { .. } => {
                    vk::COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT
                },
            };

            a | b
        };

        let (rp, sp) = if let Kind::SecondaryRenderPass { ref subpass, .. } = kind {
            (subpass.render_pass().inner().internal_object(), subpass.index())
        } else {
            (0, 0)
        };

        let framebuffer = if let Kind::SecondaryRenderPass { ref subpass, framebuffer: Some(ref framebuffer) } = kind {
            // TODO: restore check
            //assert!(framebuffer.is_compatible_with(subpass.render_pass()));     // TODO: proper error
            FramebufferAbstract::inner(&framebuffer).internal_object()
        } else {
            0
        };

        let inheritance = vk::CommandBufferInheritanceInfo {
            sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
            pNext: ptr::null(),
            renderPass: rp,
            subpass: sp,
            framebuffer: framebuffer,
            occlusionQueryEnable: 0,            // TODO:
            queryFlags: 0,          // TODO:
            pipelineStatistics: 0,          // TODO:
        };

        let infos = vk::CommandBufferBeginInfo {
            sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: ptr::null(),
            flags: vk_flags,
            pInheritanceInfo: &inheritance,
        };

        try!(check_errors(vk.BeginCommandBuffer(cmd, &infos)));

        Ok(UnsafeCommandBufferBuilder {
            cmd: Some(alloc),
            device: device.clone(),
            flags: flags,
            secondary_cb: match kind {
                Kind::Primary => false,
                Kind::Secondary | Kind::SecondaryRenderPass { .. } => true,
            },
        })
    }
}

unsafe impl<P> DeviceOwned for UnsafeCommandBufferBuilder<P> where P: CommandPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<P> CommandBufferBuilder for UnsafeCommandBufferBuilder<P> where P: CommandPool {
    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.cmd.as_ref().unwrap().queue_family()
    }
}

unsafe impl<P> VulkanObject for UnsafeCommandBufferBuilder<P> where P: CommandPool {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.cmd.as_ref().unwrap().inner().internal_object()
    }
}

unsafe impl<P> CommandBufferBuild for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBuffer<P>;
    type Err = OomError;

    #[inline]
    fn build(mut self) -> Result<Self::Out, OomError> {
        unsafe {
            let cmd = self.cmd.take().unwrap();
            let vk = self.device.pointers();
            try!(check_errors(vk.EndCommandBuffer(cmd.inner().internal_object())));

            Ok(UnsafeCommandBuffer {
                cmd: cmd.into_alloc(),
                device: self.device.clone(),
                flags: self.flags,
                already_submitted: AtomicBool::new(false),
                secondary_cb: self.secondary_cb
            })
        }
    }
}

/// Command buffer that has been built.
///
/// Doesn't perform any synchronization and doesn't keep the object it uses alive.
// TODO: change P parameter to be a CommandPoolAlloc
pub struct UnsafeCommandBuffer<P> where P: CommandPool {
    // The Vulkan command buffer.
    cmd: P::Alloc,

    // Device that owns the command buffer.
    // TODO: necessary?
    device: Arc<Device>,

    // Flags that were used at creation.
    // TODO: necessary?
    flags: Flags,

    // True if the command buffer has always been submitted once. Only relevant if `flags` is
    // `OneTimeSubmit`.
    already_submitted: AtomicBool,

    // True if this command buffer belongs to a secondary pool - needed for Drop
    secondary_cb: bool
}

unsafe impl<P> CommandBuffer for UnsafeCommandBuffer<P> where P: CommandPool {
    type Pool = P;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<P> {
        self
    }

    #[inline]
    fn submit_check(&self, _: &GpuFuture, _: &Queue) -> Result<(), CommandBufferExecError> {
        // Not our job to check.
        Ok(())
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        Err(AccessCheckError::Unknown)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: Layout, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        Err(AccessCheckError::Unknown)
    }
}

unsafe impl<P> DeviceOwned for UnsafeCommandBuffer<P> where P: CommandPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<P> VulkanObject for UnsafeCommandBuffer<P> where P: CommandPool {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.cmd.inner().internal_object()
    }
}
