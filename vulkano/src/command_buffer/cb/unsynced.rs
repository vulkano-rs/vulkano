// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use buffer::Buffer;
use command_buffer::pool::AllocatedCommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use command_buffer::CommandsListSinkCaller;
use command_buffer::DynamicState;
use command_buffer::RawCommandBufferPrototype;
use device::Device;
use framebuffer::EmptySinglePassRenderPassDesc;
use framebuffer::RenderPass;
use framebuffer::RenderPassRef;
use framebuffer::Framebuffer;
use framebuffer::Subpass;
use framebuffer::traits::FramebufferRef;
use image::Layout;
use image::Image;
use sync::AccessFlagBits;
use sync::PipelineStages;

use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Determines the kind of command buffer that we want to create.
#[derive(Debug, Clone)]
pub enum Kind<'a, R, F: 'a> {
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
        framebuffer: Option<&'a F>,
    },
}

impl<'a> Kind<'a, RenderPass<EmptySinglePassRenderPassDesc>, Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>> {
    /// Equivalent to `Kind::Primary`.
    ///
    /// > **Note**: If you use `let kind = Kind::Primary;` in your code, you will probably get a
    /// > compilation error because the Rust compiler couldn't determine the template parameters
    /// > of `Kind`. To solve that problem in an easy way you can use this function instead.
    #[inline]
    pub fn primary() -> Kind<'a, RenderPass<EmptySinglePassRenderPassDesc>, Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>> {
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

pub struct UnsyncedCommandBuffer<L, P> where P: CommandPool {
    // The Vulkan command buffer.
    cmd: vk::CommandBuffer,

    // Device that owns the command buffer.
    device: Arc<Device>,

    // Pool that owns the command buffer.
    pool: P::Finished,

    // Flags that were used at creation.
    flags: Flags,

    // True if the command buffer has always been submitted once. Only relevant if `flags` is
    // `OneTimeSubmit`.
    already_submitted: AtomicBool,

    // True if we are a secondary command buffer.
    secondary_cb: bool,

    // The commands list. Holds resources of the resources list alive. 
    commands_list: L,
}

impl<L, P> UnsyncedCommandBuffer<L, P> where L: CommandsList, P: CommandPool {
    /// Creates a new builder.
    pub unsafe fn new<R, F>(list: L, pool: P, kind: Kind<R, F>, flags: Flags)
                            -> Result<UnsyncedCommandBuffer<L, P>, OomError>
        where R: RenderPassRef, F: FramebufferRef
    {
        let secondary = match kind {
            Kind::Primary => false,
            Kind::Secondary | Kind::SecondaryRenderPass { .. } => true,
        };

        let cmd = try!(pool.alloc(secondary, 1)).next().unwrap();
        
        match UnsyncedCommandBuffer::already_allocated(list, pool, cmd, kind, flags) {
            Ok(cmd) => Ok(cmd),
            Err(err) => {
                // FIXME: uncomment this and solve the fact that `pool` has been moved
                //unsafe { pool.free(secondary, Some(cmd.into()).into_iter()) };
                Err(err)
            },
        }
    }

    /// Creates a new command buffer builder from an already-allocated command buffer.
    ///
    /// # Safety
    ///
    /// - The allocated command buffer must belong to the pool and must not be used anywhere else
    ///   in the code for the duration of this command buffer.
    ///
    pub unsafe fn already_allocated<R, F>(list: L, pool: P, cmd: AllocatedCommandBuffer,
                                          kind: Kind<R, F>, flags: Flags)
                                          -> Result<UnsyncedCommandBuffer<L, P>, OomError>
        where R: RenderPassRef, F: FramebufferRef
    {
        let device = pool.device().clone();
        let vk = device.pointers();
        let cmd = cmd.internal_object();

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
            framebuffer.inner().internal_object()
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

        {
            let mut builder = RawCommandBufferPrototype {
                device: device.clone(),
                command_buffer: Some(cmd),
                current_state: DynamicState::none(),
                bound_graphics_pipeline: 0,
                bound_compute_pipeline: 0,
                bound_index_buffer: (0, 0, 0),
                marker: PhantomData,
            };

            list.append(&mut Sink(&mut builder, &device));
        };

        try!(check_errors(vk.EndCommandBuffer(cmd)));

        Ok(UnsyncedCommandBuffer {
            device: device.clone(),
            pool: pool.finish(),
            cmd: cmd,
            flags: flags,
            secondary_cb: match kind {
                Kind::Primary => false,
                Kind::Secondary | Kind::SecondaryRenderPass { .. } => true,
            },
            already_submitted: AtomicBool::new(false),
            commands_list: list,
        })
    }

    /// Returns the device used to create this command buffer.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the list of commands of this command buffer.
    ///
    /// > **Note**: It is important that this getter is not used to modify the list of commands
    /// > with interior mutability so that `append` returns something different. Doing so is
    /// > unsafe. However this function is not unsafe, because this corner case is already covered
    /// > by the unsafetiness of the `CommandsList` trait.
    #[inline]
    pub fn commands_list(&self) -> &L {
        &self.commands_list
    }
}

unsafe impl<L, P> VulkanObject for UnsyncedCommandBuffer<L, P>
    where P: CommandPool
{
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.cmd
    }
}

// Helper object for UnsyncedCommandBuffer. Implementation detail.
struct Sink<'a>(&'a mut RawCommandBufferPrototype<'a>, &'a Arc<Device>);
impl<'a> CommandsListSink<'a> for Sink<'a> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.1
    }

    #[inline]
    fn add_command(&mut self, f: Box<CommandsListSinkCaller<'a> + 'a>) {
        f.call(self.0)
    }

    #[inline]
    fn add_buffer_transition(&mut self, _: &Buffer, _: usize, _: usize, _: bool,
                             _: PipelineStages, _: AccessFlagBits)
    {
    }

    #[inline]
    fn add_image_transition(&mut self, _: &Image, _: u32, _: u32, _: u32, _: u32,
                            _: bool, _: Layout, _: PipelineStages, _: AccessFlagBits)
    {
    }

    #[inline]
    fn add_image_transition_notification(&mut self, _: &Image, _: u32, _: u32, _: u32,
                                         _: u32, _: Layout, _: PipelineStages, _: AccessFlagBits)
    {
    }
}
