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
        pool::CommandPoolBuilderAlloc,
        synced::{Command, KeyTy, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, CommandBufferExecError,
        CommandBufferInheritanceRenderPassInfo, CommandBufferUsage, ExecuteCommandsError,
        PrimaryAutoCommandBuffer, SecondaryCommandBuffer, SubpassContents,
    },
    image::ImageLayout,
    query::QueryType,
    SafeDeref, VulkanObject,
};
use smallvec::SmallVec;

/// # Commands to execute a secondary command buffer inside a primary command buffer.
///
/// These commands can be called on any queue that can execute the commands recorded in the
/// secondary command buffer.
impl<P> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>
where
    P: CommandPoolBuilderAlloc,
{
    /// Adds a command that executes a secondary command buffer.
    ///
    /// If the `flags` that `command_buffer` was created with are more restrictive than those of
    /// `self`, then `self` will be restricted to match. E.g. executing a secondary command buffer
    /// with `Flags::OneTimeSubmit` will set `self`'s flags to `Flags::OneTimeSubmit` also.
    pub fn execute_commands<C>(
        &mut self,
        command_buffer: C,
    ) -> Result<&mut Self, ExecuteCommandsError>
    where
        C: SecondaryCommandBuffer + 'static,
    {
        self.check_command_buffer(&command_buffer)?;
        let secondary_usage = command_buffer.inner().usage();

        unsafe {
            let mut builder = self.inner.execute_commands();
            builder.add(command_buffer);
            builder.submit()?;
        }

        // Secondary command buffer could leave the primary in any state.
        self.inner.reset_state();

        // If the secondary is non-concurrent or one-time use, that restricts the primary as well.
        self.usage = std::cmp::min(self.usage, secondary_usage);

        Ok(self)
    }

    /// Adds a command that multiple secondary command buffers in a vector.
    ///
    /// This requires that the secondary command buffers do not have resource conflicts; an error
    /// will be returned if there are any. Use `execute_commands` if you want to ensure that
    /// resource conflicts are automatically resolved.
    // TODO ^ would be nice if this just worked without errors
    pub fn execute_commands_from_vec<C>(
        &mut self,
        command_buffers: Vec<C>,
    ) -> Result<&mut Self, ExecuteCommandsError>
    where
        C: SecondaryCommandBuffer + 'static,
    {
        for command_buffer in &command_buffers {
            self.check_command_buffer(command_buffer)?;
        }

        let mut secondary_usage = CommandBufferUsage::SimultaneousUse; // Most permissive usage
        unsafe {
            let mut builder = self.inner.execute_commands();
            for command_buffer in command_buffers {
                secondary_usage = std::cmp::min(secondary_usage, command_buffer.inner().usage());
                builder.add(command_buffer);
            }
            builder.submit()?;
        }

        // Secondary command buffer could leave the primary in any state.
        self.inner.reset_state();

        // If the secondary is non-concurrent or one-time use, that restricts the primary as well.
        self.usage = std::cmp::min(self.usage, secondary_usage);

        Ok(self)
    }

    // Helper function for execute_commands
    fn check_command_buffer<C>(
        &self,
        command_buffer: &C,
    ) -> Result<(), AutoCommandBufferBuilderContextError>
    where
        C: SecondaryCommandBuffer + 'static,
    {
        if let Some(render_pass) = &command_buffer.inheritance_info().render_pass {
            self.ensure_inside_render_pass_secondary(render_pass)?;
        } else {
            self.ensure_outside_render_pass()?;
        }

        for state in self.query_state.values() {
            match state.ty {
                QueryType::Occlusion => match command_buffer.inheritance_info().occlusion_query {
                    Some(inherited_flags) => {
                        let inherited_flags = ash::vk::QueryControlFlags::from(inherited_flags);
                        let state_flags = ash::vk::QueryControlFlags::from(state.flags);

                        if inherited_flags & state_flags != state_flags {
                            return Err(AutoCommandBufferBuilderContextError::QueryNotInherited);
                        }
                    }
                    None => return Err(AutoCommandBufferBuilderContextError::QueryNotInherited),
                },
                QueryType::PipelineStatistics(state_flags) => {
                    let inherited_flags = command_buffer.inheritance_info().query_statistics_flags;
                    let inherited_flags =
                        ash::vk::QueryPipelineStatisticFlags::from(inherited_flags);
                    let state_flags = ash::vk::QueryPipelineStatisticFlags::from(state_flags);

                    if inherited_flags & state_flags != state_flags {
                        return Err(AutoCommandBufferBuilderContextError::QueryNotInherited);
                    }
                }
                _ => (),
            }
        }

        Ok(())
    }

    #[inline]
    fn ensure_inside_render_pass_secondary(
        &self,
        render_pass: &CommandBufferInheritanceRenderPassInfo,
    ) -> Result<(), AutoCommandBufferBuilderContextError> {
        let render_pass_state = self
            .render_pass_state
            .as_ref()
            .ok_or(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass)?;

        if render_pass_state.contents != SubpassContents::SecondaryCommandBuffers {
            return Err(AutoCommandBufferBuilderContextError::WrongSubpassType);
        }

        // Subpasses must be the same.
        if render_pass.subpass.index() != render_pass_state.subpass.index() {
            return Err(AutoCommandBufferBuilderContextError::WrongSubpassIndex);
        }

        // Render passes must be compatible.
        if !render_pass
            .subpass
            .render_pass()
            .is_compatible_with(render_pass_state.subpass.render_pass())
        {
            return Err(AutoCommandBufferBuilderContextError::IncompatibleRenderPass);
        }

        // Framebuffer, if present on the secondary command buffer, must be the
        // same as the one in the current render pass.
        if let Some(framebuffer) = &render_pass.framebuffer {
            if framebuffer.internal_object() != render_pass_state.framebuffer {
                return Err(AutoCommandBufferBuilderContextError::IncompatibleFramebuffer);
            }
        }

        Ok(())
    }
}

impl SyncCommandBufferBuilder {
    /// Starts the process of executing secondary command buffers. Returns an intermediate struct
    /// which can be used to add the command buffers.
    #[inline]
    pub unsafe fn execute_commands(&mut self) -> SyncCommandBufferBuilderExecuteCommands {
        SyncCommandBufferBuilderExecuteCommands {
            builder: self,
            inner: Vec::new(),
        }
    }
}

/// Prototype for a `vkCmdExecuteCommands`.
pub struct SyncCommandBufferBuilderExecuteCommands<'a> {
    builder: &'a mut SyncCommandBufferBuilder,
    inner: Vec<Box<dyn SecondaryCommandBuffer>>,
}

impl<'a> SyncCommandBufferBuilderExecuteCommands<'a> {
    /// Adds a command buffer to the list.
    #[inline]
    pub fn add<C>(&mut self, command_buffer: C)
    where
        C: SecondaryCommandBuffer + 'static,
    {
        self.inner.push(Box::new(command_buffer));
    }

    #[inline]
    pub unsafe fn submit(self) -> Result<(), SyncCommandBufferBuilderError> {
        struct DropUnlock(Box<dyn SecondaryCommandBuffer>);
        impl std::ops::Deref for DropUnlock {
            type Target = Box<dyn SecondaryCommandBuffer>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        unsafe impl SafeDeref for DropUnlock {}

        impl Drop for DropUnlock {
            fn drop(&mut self) {
                unsafe {
                    self.unlock();
                }
            }
        }

        struct Cmd(Vec<DropUnlock>);

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdExecuteCommands"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                let mut execute = UnsafeCommandBufferBuilderExecuteCommands::new();
                self.0
                    .iter()
                    .for_each(|cbuf| execute.add_raw(cbuf.inner().internal_object()));
                out.execute_commands(execute);
            }
        }

        let resources = {
            let mut resources = Vec::new();
            for (cbuf_num, cbuf) in self.inner.iter().enumerate() {
                for buf_num in 0..cbuf.num_buffers() {
                    resources.push((
                        KeyTy::Buffer(cbuf.buffer(buf_num).unwrap().0.clone()),
                        format!("Buffer bound to secondary command buffer {}", cbuf_num).into(),
                        Some((
                            cbuf.buffer(buf_num).unwrap().1,
                            ImageLayout::Undefined,
                            ImageLayout::Undefined,
                        )),
                    ));
                }
                for img_num in 0..cbuf.num_images() {
                    let (_, memory, start_layout, end_layout) = cbuf.image(img_num).unwrap();
                    resources.push((
                        KeyTy::Image(cbuf.image(img_num).unwrap().0.clone()),
                        format!("Image bound to secondary command buffer {}", cbuf_num).into(),
                        Some((memory, start_layout, end_layout)),
                    ));
                }
            }
            resources
        };

        self.builder.append_command(
            Cmd(self
                .inner
                .into_iter()
                .map(|cbuf| {
                    cbuf.lock_record()?;
                    Ok(DropUnlock(cbuf))
                })
                .collect::<Result<Vec<_>, CommandBufferExecError>>()?),
            resources,
        )?;

        Ok(())
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdExecuteCommands` on the builder.
    ///
    /// Does nothing if the list of command buffers is empty, as it would be a no-op and isn't a
    /// valid usage of the command anyway.
    #[inline]
    pub unsafe fn execute_commands(&mut self, cbs: UnsafeCommandBufferBuilderExecuteCommands) {
        if cbs.raw_cbs.is_empty() {
            return;
        }

        let fns = self.device.fns();
        fns.v1_0
            .cmd_execute_commands(self.handle, cbs.raw_cbs.len() as u32, cbs.raw_cbs.as_ptr());
    }
}

/// Prototype for a `vkCmdExecuteCommands`.
pub struct UnsafeCommandBufferBuilderExecuteCommands {
    // Raw handles of the command buffers to execute.
    raw_cbs: SmallVec<[ash::vk::CommandBuffer; 4]>,
}

impl UnsafeCommandBufferBuilderExecuteCommands {
    /// Builds a new empty list.
    #[inline]
    pub fn new() -> UnsafeCommandBufferBuilderExecuteCommands {
        UnsafeCommandBufferBuilderExecuteCommands {
            raw_cbs: SmallVec::new(),
        }
    }

    /// Adds a command buffer to the list.
    #[inline]
    pub fn add<C>(&mut self, cb: &C)
    where
        C: ?Sized + SecondaryCommandBuffer,
    {
        // TODO: debug assert that it is a secondary command buffer?
        self.raw_cbs.push(cb.inner().internal_object());
    }

    /// Adds a command buffer to the list.
    #[inline]
    pub unsafe fn add_raw(&mut self, cb: ash::vk::CommandBuffer) {
        self.raw_cbs.push(cb);
    }
}
