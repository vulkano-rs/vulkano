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
use device::Device;
use device::DeviceOwned;
use instance::QueueFamily;

/// Layer around a command buffer builder that checks whether the commands can be executed in the
/// given context related to render passes.
///
/// What is checked exactly:
///
/// - When adding a command that can only be executed within a render pass or outside of a render
///   pass, checks that we are within or outside of a render pass.
/// - When leaving the render pass or going to the next subpass, makes sure that the number of
///   subpasses of the current render pass is respected.
/// - When binding a graphics pipeline or drawing, makes sure that the pipeline is valid for the
///   current render pass.
///
pub struct ContextCheckLayer<I> {
    // Inner command buffer builder.
    inner: I,
    // True if we are currently inside a render pass.
    inside_render_pass: bool,
    // True if entering/leaving a render pass or going to the next subpass is allowed.
    allow_render_pass_ops: bool,
}

impl<I> ContextCheckLayer<I> {
    /// Builds a new `ContextCheckLayer`.
    ///
    /// If `allow_render_pass_ops` is true, then entering/leaving a render pass or going to the
    /// next subpass is allowed by the layer.
    ///
    /// If `inside_render_pass` is true, then the builder is currently inside a render pass.
    ///
    /// Note that this layer will only protect you if you pass correct values in this constructor.
    /// It is not unsafe to pass wrong values, but if you do so then the layer will be inefficient
    /// as a safety tool.
    #[inline]
    pub fn new(inner: I, inside_render_pass: bool, allow_render_pass_ops: bool)
               -> ContextCheckLayer<I>
    {
        ContextCheckLayer {
            inner: inner,
            inside_render_pass: inside_render_pass,
            allow_render_pass_ops: allow_render_pass_ops,
        }
    }

    /// Destroys the layer and returns the underlying command buffer.
    #[inline]
    pub fn into_inner(self) -> I {
        self.inner
    }
}

unsafe impl<I, O, E> CommandBufferBuild for ContextCheckLayer<I>
    where I: CommandBufferBuild<Out = O, Err = E>
{
    type Out = O;
    type Err = E;

    #[inline]
    fn build(self) -> Result<O, E> {
        self.inner.build()
    }
}

unsafe impl<I> DeviceOwned for ContextCheckLayer<I>
    where I: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I> CommandBufferBuilder for ContextCheckLayer<I>
    where I: CommandBufferBuilder
{
    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.inner.queue_family()
    }
}

// TODO:
// impl!((C), commands_raw::CmdExecuteCommands<C>);

// FIXME: must also check that a pipeline's render pass matches the render pass

// FIXME:
// > If the variable multisample rate feature is not supported, pipeline is a graphics pipeline,
// > the current subpass has no attachments, and this is not the first call to this function with
// > a graphics pipeline after transitioning to the current subpass, then the sample count
// > specified by this pipeline must match that set in the previous pipeline

macro_rules! impl_always {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for ContextCheckLayer<I>
            where I: AddCommand<$cmd, Out = O>
        {
            type Out = ContextCheckLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Result<Self::Out, CommandAddError> {
                Ok(ContextCheckLayer {
                    inner: self.inner.add(command)?,
                    inside_render_pass: self.inside_render_pass,
                    allow_render_pass_ops: self.allow_render_pass_ops,
                })
            }
        }
    }
}

impl_always!((S, Pl), commands_raw::CmdBindDescriptorSets<S, Pl>);
impl_always!((B), commands_raw::CmdBindIndexBuffer<B>);
impl_always!((Pl), commands_raw::CmdBindPipeline<Pl>);
impl_always!((V), commands_raw::CmdBindVertexBuffers<V>);
impl_always!((Pc, Pl), commands_raw::CmdPushConstants<Pc, Pl>);
impl_always!((), commands_raw::CmdSetState);

macro_rules! impl_inside_only {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for ContextCheckLayer<I>
            where I: AddCommand<$cmd, Out = O>
        {
            type Out = ContextCheckLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Result<Self::Out, CommandAddError> {
                if !self.inside_render_pass {
                    return Err(CommandAddError::ForbiddenOutsideRenderPass);
                }

                Ok(ContextCheckLayer {
                    inner: self.inner.add(command)?,
                    inside_render_pass: self.inside_render_pass,
                    allow_render_pass_ops: self.allow_render_pass_ops,
                })
            }
        }
    }
}

impl_inside_only!((), commands_raw::CmdClearAttachments);
impl_inside_only!((), commands_raw::CmdDrawIndexedRaw);
impl_inside_only!((B), commands_raw::CmdDrawIndirectRaw<B>);
impl_inside_only!((), commands_raw::CmdDrawRaw);

macro_rules! impl_outside_only {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for ContextCheckLayer<I>
            where I: AddCommand<$cmd, Out = O>
        {
            type Out = ContextCheckLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Result<Self::Out, CommandAddError> {
                if self.inside_render_pass {
                    return Err(CommandAddError::ForbiddenInsideRenderPass);
                }

                Ok(ContextCheckLayer {
                    inner: self.inner.add(command)?,
                    inside_render_pass: self.inside_render_pass,
                    allow_render_pass_ops: self.allow_render_pass_ops,
                })
            }
        }
    }
}

impl_outside_only!((S, D), commands_raw::CmdBlitImage<S, D>);
impl_outside_only!((S, D), commands_raw::CmdCopyBuffer<S, D>);
impl_outside_only!((S, D), commands_raw::CmdCopyBufferToImage<S, D>);
impl_outside_only!((S, D), commands_raw::CmdCopyImage<S, D>);
impl_outside_only!((), commands_raw::CmdDispatchRaw);
impl_outside_only!((B), commands_raw::CmdFillBuffer<B>);
impl_outside_only!((S, D), commands_raw::CmdResolveImage<S, D>);
impl_outside_only!((), commands_raw::CmdSetEvent);
impl_outside_only!((B, D), commands_raw::CmdUpdateBuffer<B, D>);

unsafe impl<'a, I, O, Rp, F> AddCommand<commands_raw::CmdBeginRenderPass<Rp, F>> for ContextCheckLayer<I>
    where I: AddCommand<commands_raw::CmdBeginRenderPass<Rp, F>, Out = O>
{
    type Out = ContextCheckLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdBeginRenderPass<Rp, F>) -> Result<Self::Out, CommandAddError> {
        if self.inside_render_pass {
            return Err(CommandAddError::ForbiddenInsideRenderPass);
        }
        
        if !self.allow_render_pass_ops {
            return Err(CommandAddError::ForbiddenInSecondaryCommandBuffer);
        }

        Ok(ContextCheckLayer {
            inner: self.inner.add(command)?,
            inside_render_pass: true,
            allow_render_pass_ops: true,
        })
    }
}

unsafe impl<'a, I, O> AddCommand<commands_raw::CmdNextSubpass> for ContextCheckLayer<I>
    where I: AddCommand<commands_raw::CmdNextSubpass, Out = O>
{
    type Out = ContextCheckLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdNextSubpass) -> Result<Self::Out, CommandAddError> {
        if !self.inside_render_pass {
            return Err(CommandAddError::ForbiddenOutsideRenderPass);
        }

        if !self.allow_render_pass_ops {
            return Err(CommandAddError::ForbiddenInSecondaryCommandBuffer);
        }

        // FIXME: check number of subpasses

        Ok(ContextCheckLayer {
            inner: self.inner.add(command)?,
            inside_render_pass: true,
            allow_render_pass_ops: true,
        })
    }
}

unsafe impl<'a, I, O> AddCommand<commands_raw::CmdEndRenderPass> for ContextCheckLayer<I>
    where I: AddCommand<commands_raw::CmdEndRenderPass, Out = O>
{
    type Out = ContextCheckLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdEndRenderPass) -> Result<Self::Out, CommandAddError> {
        if !self.inside_render_pass {
            return Err(CommandAddError::ForbiddenOutsideRenderPass);
        }

        if !self.allow_render_pass_ops {
            return Err(CommandAddError::ForbiddenInSecondaryCommandBuffer);
        }

        // FIXME: check number of subpasses

        Ok(ContextCheckLayer {
            inner: self.inner.add(command)?,
            inside_render_pass: false,
            allow_render_pass_ops: true,
        })
    }
}
