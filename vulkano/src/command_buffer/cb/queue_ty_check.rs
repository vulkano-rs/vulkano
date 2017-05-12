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
use command_buffer::CommandBuffer;
use command_buffer::commands_raw;
use device::Device;
use device::DeviceOwned;
use instance::QueueFamily;
use VulkanObject;

/// Layer around a command buffer builder that checks whether the commands added to it match the
/// type of the queue family of the underlying builder.
///
/// Commands that perform graphical or compute operations can only be executed on queue families
/// that support graphical or compute operations. This is what this layer verifies.
pub struct QueueTyCheckLayer<I> {
    inner: I,
}

impl<I> QueueTyCheckLayer<I> {
    /// Builds a new `QueueTyCheckLayer`.
    #[inline]
    pub fn new(inner: I) -> QueueTyCheckLayer<I> {
        QueueTyCheckLayer {
            inner: inner,
        }
    }

    /// Destroys the layer and returns the underlying command buffer.
    #[inline]
    pub fn into_inner(self) -> I {
        self.inner
    }
}

unsafe impl<I> DeviceOwned for QueueTyCheckLayer<I>
    where I: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I> CommandBufferBuilder for QueueTyCheckLayer<I>
    where I: CommandBufferBuilder
{
    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.inner.queue_family()
    }
}

unsafe impl<I, O, E> CommandBufferBuild for QueueTyCheckLayer<I>
    where I: CommandBufferBuild<Out = O, Err = E>
{
    type Out = O;
    type Err = E;

    #[inline]
    fn build(self) -> Result<O, E> {
        self.inner.build()
    }
}

macro_rules! q_ty_impl_always {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for QueueTyCheckLayer<I>
            where I: CommandBufferBuilder + AddCommand<$cmd, Out = O>
        {
            type Out = QueueTyCheckLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Result<Self::Out, CommandAddError> {
                Ok(QueueTyCheckLayer {
                    inner: self.inner.add(command)?,
                })
            }
        }
    }
}

q_ty_impl_always!((S, D), commands_raw::CmdCopyBuffer<S, D>);
q_ty_impl_always!((S, D), commands_raw::CmdCopyBufferToImage<S, D>);
q_ty_impl_always!((S, D), commands_raw::CmdCopyImage<S, D>);
q_ty_impl_always!((B), commands_raw::CmdFillBuffer<B>);
q_ty_impl_always!((B, D), commands_raw::CmdUpdateBuffer<B, D>);

macro_rules! q_ty_impl_graphics {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for QueueTyCheckLayer<I>
            where I: CommandBufferBuilder + AddCommand<$cmd, Out = O>
        {
            type Out = QueueTyCheckLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Result<Self::Out, CommandAddError> {
                if !self.supports_graphics() {
                    return Err(CommandAddError::GraphicsOperationsNotSupported);
                }

                Ok(QueueTyCheckLayer {
                    inner: self.inner.add(command)?,
                })
            }
        }
    }
}

q_ty_impl_graphics!((Rp, F), commands_raw::CmdBeginRenderPass<Rp, F>);
q_ty_impl_graphics!((B), commands_raw::CmdBindIndexBuffer<B>);
q_ty_impl_graphics!((V), commands_raw::CmdBindVertexBuffers<V>);
q_ty_impl_graphics!((S, D), commands_raw::CmdBlitImage<S, D>);
q_ty_impl_graphics!((), commands_raw::CmdClearAttachments);
q_ty_impl_graphics!((), commands_raw::CmdDrawIndexedRaw);
q_ty_impl_graphics!((B), commands_raw::CmdDrawIndirectRaw<B>);
q_ty_impl_graphics!((), commands_raw::CmdDrawRaw);
q_ty_impl_graphics!((), commands_raw::CmdEndRenderPass);
q_ty_impl_graphics!((), commands_raw::CmdNextSubpass);
q_ty_impl_graphics!((S, D), commands_raw::CmdResolveImage<S, D>);

macro_rules! q_ty_impl_compute {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for QueueTyCheckLayer<I>
            where I: CommandBufferBuilder + AddCommand<$cmd, Out = O>
        {
            type Out = QueueTyCheckLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Result<Self::Out, CommandAddError> {
                if !self.supports_compute() {
                    return Err(CommandAddError::ComputeOperationsNotSupported);
                }

                Ok(QueueTyCheckLayer {
                    inner: self.inner.add(command)?,
                })
            }
        }
    }
}

q_ty_impl_compute!((), commands_raw::CmdDispatchRaw);

macro_rules! q_ty_impl_graphics_or_compute {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for QueueTyCheckLayer<I>
            where I: CommandBufferBuilder + AddCommand<$cmd, Out = O>
        {
            type Out = QueueTyCheckLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Result<Self::Out, CommandAddError> {
                assert!(self.supports_graphics() || self.supports_compute());      // TODO: proper error?
                Ok(QueueTyCheckLayer {
                    inner: self.inner.add(command)?,
                })
            }
        }
    }
}

q_ty_impl_graphics_or_compute!((Pc, Pl), commands_raw::CmdPushConstants<Pc, Pl>);
q_ty_impl_graphics_or_compute!((), commands_raw::CmdSetEvent);
q_ty_impl_graphics_or_compute!((), commands_raw::CmdSetState);

unsafe impl<I, O, Pl> AddCommand<commands_raw::CmdBindPipeline<Pl>> for QueueTyCheckLayer<I>
    where I: CommandBufferBuilder + AddCommand<commands_raw::CmdBindPipeline<Pl>, Out = O>
{
    type Out = QueueTyCheckLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdBindPipeline<Pl>) -> Result<Self::Out, CommandAddError> {
        if command.is_graphics() {
            if !self.supports_graphics() {
                return Err(CommandAddError::GraphicsOperationsNotSupported);
            }
        } else {
            if !self.supports_compute() {
                return Err(CommandAddError::ComputeOperationsNotSupported);
            }
        }

        Ok(QueueTyCheckLayer {
            inner: self.inner.add(command)?,
        })
    }
}

unsafe impl<I, O, S, Pl> AddCommand<commands_raw::CmdBindDescriptorSets<S, Pl>> for QueueTyCheckLayer<I>
    where I: CommandBufferBuilder + AddCommand<commands_raw::CmdBindDescriptorSets<S, Pl>, Out = O>
{
    type Out = QueueTyCheckLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdBindDescriptorSets<S, Pl>) -> Result<Self::Out, CommandAddError> {
        if command.is_graphics() {
            if !self.supports_graphics() {
                return Err(CommandAddError::GraphicsOperationsNotSupported);
            }
        } else {
            if !self.supports_compute() {
                return Err(CommandAddError::ComputeOperationsNotSupported);
            }
        }

        Ok(QueueTyCheckLayer {
            inner: self.inner.add(command)?,
        })
    }
}

unsafe impl<I, O, C> AddCommand<commands_raw::CmdExecuteCommands<C>> for QueueTyCheckLayer<I>
    where I: CommandBufferBuilder + AddCommand<commands_raw::CmdExecuteCommands<C>, Out = O>,
          C: CommandBuffer
{
    type Out = QueueTyCheckLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdExecuteCommands<C>) -> Result<Self::Out, CommandAddError> {
        // Note that safety rules guarantee that the secondary command buffer belongs to the same
        // device as ourselves. Therefore this assert is only a debug assert.
        debug_assert_eq!(command.command_buffer().queue_family().physical_device().internal_object(),
                         self.queue_family().physical_device().internal_object());

        if command.command_buffer().queue_family().id() != self.queue_family().id() {
            return Err(CommandAddError::QueueFamilyMismatch);
        }

        Ok(QueueTyCheckLayer {
            inner: self.inner.add(command)?,
        })
    }
}
