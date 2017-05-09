// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use command_buffer::cb::AddCommand;
use command_buffer::cb::CommandBufferBuild;
use command_buffer::CommandAddError;
use command_buffer::CommandBufferBuilder;
use command_buffer::commands_raw;
use command_buffer::DynamicState;
use device::Device;
use device::DeviceOwned;
use instance::QueueFamily;
use VulkanObject;
use vk;

/// Layer around a command buffer builder that caches the current state of the command buffer and
/// avoids redundant state changes.
///
/// For example if you add a command that sets the current vertex buffer, then later another
/// command that sets the current vertex buffer to the same value, then the second one will be
/// discarded by this layer.
///
/// As a general rule there's no reason not to use this layer unless you know that your commands
/// are already optimized in this regard.
///
/// # Safety
///
/// This layer expects that the commands passed to it all belong to the same device.
///
/// Since this layer can potentially optimize out some commands, a mismatch between devices could
/// potentially go undetected if it is checked in a lower layer.
pub struct StateCacheLayer<I> {
    // The inner builder that will actually execute the stuff.
    inner: I,
    // The dynamic state to synchronize with `CmdSetState`.
    dynamic_state: DynamicState,
    // The compute pipeline currently bound. 0 if nothing bound.
    compute_pipeline: vk::Pipeline,
    // The graphics pipeline currently bound. 0 if nothing bound.
    graphics_pipeline: vk::Pipeline,
    // The latest bind vertex buffers command.
    vertex_buffers: Option<commands_raw::CmdBindVertexBuffersHash>,
}

impl<I> StateCacheLayer<I> {
    /// Builds a new `StateCacheLayer`.
    ///
    /// It is safe to start caching at any point of the construction of a command buffer.
    #[inline]
    pub fn new(inner: I) -> StateCacheLayer<I> {
        StateCacheLayer {
            inner: inner,
            dynamic_state: DynamicState::none(),
            compute_pipeline: 0,
            graphics_pipeline: 0,
            vertex_buffers: None,
        }
    }

    /// Destroys the layer and returns the underlying command buffer.
    #[inline]
    pub fn into_inner(self) -> I {
        self.inner
    }
}

unsafe impl<I> DeviceOwned for StateCacheLayer<I>
    where I: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I> CommandBufferBuilder for StateCacheLayer<I>
    where I: CommandBufferBuilder
{
    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.inner.queue_family()
    }
}

unsafe impl<Pl, I, O> AddCommand<commands_raw::CmdBindPipeline<Pl>> for StateCacheLayer<I>
    where I: AddCommand<commands_raw::CmdBindPipeline<Pl>, Out = O>
{
    type Out = StateCacheLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdBindPipeline<Pl>) -> Result<Self::Out, CommandAddError> {
        let raw_pipeline = command.sys().internal_object();

        let new_command = {
            if command.is_graphics() {
                if raw_pipeline == self.graphics_pipeline {
                    command.disabled()
                } else {
                    self.graphics_pipeline = raw_pipeline;
                    command
                }
            } else {
                if raw_pipeline == self.compute_pipeline {
                    command.disabled()
                } else {
                    self.compute_pipeline = raw_pipeline;
                    command
                }
            }
        };

        Ok(StateCacheLayer {
            inner: self.inner.add(new_command)?,
            dynamic_state: DynamicState::none(),
            graphics_pipeline: self.graphics_pipeline,
            compute_pipeline: self.compute_pipeline,
            vertex_buffers: self.vertex_buffers,
        })
    }
}

unsafe impl<Cb, I, O> AddCommand<commands_raw::CmdExecuteCommands<Cb>> for StateCacheLayer<I>
    where I: AddCommand<commands_raw::CmdExecuteCommands<Cb>, Out = O>
{
    type Out = StateCacheLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdExecuteCommands<Cb>) -> Result<Self::Out, CommandAddError> {
        // After a secondary command buffer is added, all states at reset to the "unknown" state.
        let new_inner = self.inner.add(command)?;

        Ok(StateCacheLayer {
            inner: new_inner,
            dynamic_state: DynamicState::none(),
            compute_pipeline: 0,
            graphics_pipeline: 0,
            vertex_buffers: None,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdSetState> for StateCacheLayer<I>
    where I: AddCommand<commands_raw::CmdSetState, Out = O>
{
    type Out = StateCacheLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdSetState) -> Result<Self::Out, CommandAddError> {
        // We need to synchronize `self.dynamic_state` with the state in `command`.
        // While doing so, we tweak `command` to erase the states that are the same as what's
        // already in `self.dynamic_state`.

        let mut command_state = command.state().clone();

        // Handle line width.
        if let Some(new_val) = command_state.line_width {
            if self.dynamic_state.line_width == Some(new_val) {
                command_state.line_width = None;
            } else {
                self.dynamic_state.line_width = Some(new_val);
            }
        }

        // TODO: missing implementations

        Ok(StateCacheLayer {
            inner: self.inner.add(commands_raw::CmdSetState::new(command.device().clone(), command_state))?,
            dynamic_state: self.dynamic_state,
            graphics_pipeline: self.graphics_pipeline,
            compute_pipeline: self.compute_pipeline,
            vertex_buffers: self.vertex_buffers,
        })
    }
}

unsafe impl<I, O, B> AddCommand<commands_raw::CmdBindVertexBuffers<B>> for StateCacheLayer<I>
    where I: AddCommand<commands_raw::CmdBindVertexBuffers<B>, Out = O>
{
    type Out = StateCacheLayer<O>;

    #[inline]
    fn add(mut self, mut command: commands_raw::CmdBindVertexBuffers<B>)
           -> Result<Self::Out, CommandAddError>
    {
        match &mut self.vertex_buffers {
            &mut Some(ref mut curr) => {
                if *curr != *command.hash() {
                    let new_hash = command.hash().clone();
                    command.diff(curr);
                    *curr = new_hash;
                }
            },
            curr @ &mut None => {
                *curr = Some(command.hash().clone());
            }
        };

        Ok(StateCacheLayer {
            inner: self.inner.add(command)?,
            dynamic_state: self.dynamic_state,
            graphics_pipeline: self.graphics_pipeline,
            compute_pipeline: self.compute_pipeline,
            vertex_buffers: self.vertex_buffers,
        })
    }
}

unsafe impl<I, O, E> CommandBufferBuild for StateCacheLayer<I>
    where I: CommandBufferBuild<Out = O, Err = E>
{
    type Out = O;
    type Err = E;

    #[inline]
    fn build(self) -> Result<O, E> {
        self.inner.build()
    }
}

macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for StateCacheLayer<I>
            where I: AddCommand<$cmd, Out = O>
        {
            type Out = StateCacheLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Result<Self::Out, CommandAddError> {
                Ok(StateCacheLayer {
                    inner: self.inner.add(command)?,
                    dynamic_state: self.dynamic_state,
                    graphics_pipeline: self.graphics_pipeline,
                    compute_pipeline: self.compute_pipeline,
                    vertex_buffers: self.vertex_buffers,
                })
            }
        }
    }
}

pass_through!((Rp, F), commands_raw::CmdBeginRenderPass<Rp, F>);
pass_through!((S, Pl), commands_raw::CmdBindDescriptorSets<S, Pl>);
pass_through!((B), commands_raw::CmdBindIndexBuffer<B>);
pass_through!((S, D), commands_raw::CmdBlitImage<S, D>);
pass_through!((), commands_raw::CmdClearAttachments);
pass_through!((S, D), commands_raw::CmdCopyBuffer<S, D>);
pass_through!((S, D), commands_raw::CmdCopyBufferToImage<S, D>);
pass_through!((S, D), commands_raw::CmdCopyImage<S, D>);
pass_through!((), commands_raw::CmdDispatchRaw);
pass_through!((), commands_raw::CmdDrawIndexedRaw);
pass_through!((B), commands_raw::CmdDrawIndirectRaw<B>);
pass_through!((), commands_raw::CmdDrawRaw);
pass_through!((), commands_raw::CmdEndRenderPass);
pass_through!((B), commands_raw::CmdFillBuffer<B>);
pass_through!((), commands_raw::CmdNextSubpass);
pass_through!((Pc, Pl), commands_raw::CmdPushConstants<Pc, Pl>);
pass_through!((S, D), commands_raw::CmdResolveImage<S, D>);
pass_through!((), commands_raw::CmdSetEvent);
pass_through!((B, D), commands_raw::CmdUpdateBuffer<B, D>);
