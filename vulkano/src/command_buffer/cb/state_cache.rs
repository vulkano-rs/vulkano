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
use command_buffer::CommandBufferBuilder;
use command_buffer::cmd;
use command_buffer::DynamicState;
use device::Device;
use device::DeviceOwned;
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
}

unsafe impl<Pl, I, O> AddCommand<cmd::CmdBindPipeline<Pl>> for StateCacheLayer<I>
    where I: AddCommand<cmd::CmdBindPipeline<Pl>, Out = O>
{
    type Out = StateCacheLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdBindPipeline<Pl>) -> Self::Out {
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

        StateCacheLayer {
            inner: self.inner.add(new_command),
            dynamic_state: DynamicState::none(),
            graphics_pipeline: self.graphics_pipeline,
            compute_pipeline: self.compute_pipeline,
        }
    }
}

unsafe impl<Cb, I, O> AddCommand<cmd::CmdExecuteCommands<Cb>> for StateCacheLayer<I>
    where I: AddCommand<cmd::CmdExecuteCommands<Cb>, Out = O>
{
    type Out = StateCacheLayer<O>;

    #[inline]
    fn add(self, command: cmd::CmdExecuteCommands<Cb>) -> Self::Out {
        // After a secondary command buffer is added, all states at reset to the "unknown" state.
        let new_inner = self.inner.add(command);

        StateCacheLayer {
            inner: new_inner,
            dynamic_state: DynamicState::none(),
            compute_pipeline: 0,
            graphics_pipeline: 0,
        }
    }
}

unsafe impl<I, O> AddCommand<cmd::CmdSetState> for StateCacheLayer<I>
    where I: AddCommand<cmd::CmdSetState, Out = O>
{
    type Out = StateCacheLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdSetState) -> Self::Out {
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

        StateCacheLayer {
            inner: self.inner.add(cmd::CmdSetState::new(command.device().clone(), command_state)),
            dynamic_state: self.dynamic_state,
            graphics_pipeline: self.graphics_pipeline,
            compute_pipeline: self.compute_pipeline,
        }
    }
}

unsafe impl<I, O> CommandBufferBuild for StateCacheLayer<I>
    where I: CommandBufferBuild<Out = O>
{
    type Out = O;

    #[inline]
    fn build(self) -> O {
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
            fn add(self, command: $cmd) -> Self::Out {
                StateCacheLayer {
                    inner: self.inner.add(command),
                    dynamic_state: self.dynamic_state,
                    graphics_pipeline: self.graphics_pipeline,
                    compute_pipeline: self.compute_pipeline,
                }
            }
        }
    }
}

pass_through!((Rp, F), cmd::CmdBeginRenderPass<Rp, F>);
pass_through!((S, Pl), cmd::CmdBindDescriptorSets<S, Pl>);
pass_through!((B), cmd::CmdBindIndexBuffer<B>);
pass_through!((V), cmd::CmdBindVertexBuffers<V>);
pass_through!((S, D), cmd::CmdBlitImage<S, D>);
pass_through!((), cmd::CmdClearAttachments);
pass_through!((S, D), cmd::CmdCopyBuffer<S, D>);
pass_through!((S, D), cmd::CmdCopyBufferToImage<S, D>);
pass_through!((S, D), cmd::CmdCopyImage<S, D>);
pass_through!((), cmd::CmdDispatchRaw);
pass_through!((), cmd::CmdDrawIndexedRaw);
pass_through!((B), cmd::CmdDrawIndirectRaw<B>);
pass_through!((), cmd::CmdDrawRaw);
pass_through!((), cmd::CmdEndRenderPass);
pass_through!((B), cmd::CmdFillBuffer<B>);
pass_through!((), cmd::CmdNextSubpass);
pass_through!((Pc, Pl), cmd::CmdPushConstants<Pc, Pl>);
pass_through!((S, D), cmd::CmdResolveImage<S, D>);
pass_through!((), cmd::CmdSetEvent);
pass_through!((B, D), cmd::CmdUpdateBuffer<'a, B, D>);
