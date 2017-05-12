// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::hash_map::Entry;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use fnv::FnvHashMap;

use buffer::BufferAccess;
use command_buffer::cb::AddCommand;
use command_buffer::cb::CommandBufferBuild;
use command_buffer::cb::UnsafeCommandBuffer;
use command_buffer::CommandAddError;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferBuilder;
use command_buffer::commands_raw;
use framebuffer::FramebufferAbstract;
use image::Layout;
use image::ImageAccess;
use instance::QueueFamily;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use sync::AccessCheckError;
use sync::AccessError;
use sync::AccessFlagBits;
use sync::PipelineStages;
use sync::GpuFuture;

/// Layers that ensures that synchronization with buffers and images between command buffers is
/// properly handled.
///
/// The following are handled:
///
/// - Return an error when submitting if the user didn't provide the guarantees for proper
///   synchronization.
///
/// - Automatically generate pipeline barriers between command buffers if necessary to handle
///   the transition between command buffers.
///
pub struct SubmitSyncBuilderLayer<I> {
    inner: I,
    resources: FnvHashMap<Key, ResourceEntry>,
}

enum Key {
    Buffer(Box<BufferAccess + Send + Sync>),
    Image(Box<ImageAccess + Send + Sync>),
    FramebufferAttachment(Box<FramebufferAbstract + Send + Sync>, u32),
}

impl Key {
    #[inline]
    fn conflicts_buffer_all(&self, buf: &BufferAccess) -> bool {
        match self {
            &Key::Buffer(ref a) => a.conflicts_buffer_all(buf),
            &Key::Image(ref a) => a.conflicts_buffer_all(buf),
            &Key::FramebufferAttachment(ref b, idx) => {
                let img = b.attachments()[idx as usize].parent();
                img.conflicts_buffer_all(buf)
            },
        }
    }

    #[inline]
    fn conflicts_image_all(&self, img: &ImageAccess) -> bool {
        match self {
            &Key::Buffer(ref a) => a.conflicts_image_all(img),
            &Key::Image(ref a) => a.conflicts_image_all(img),
            &Key::FramebufferAttachment(ref b, idx) => {
                let b = b.attachments()[idx as usize].parent();
                b.conflicts_image_all(img)
            },
        }
    }
}

impl PartialEq for Key {
    #[inline]
    fn eq(&self, other: &Key) -> bool {
        match other {
            &Key::Buffer(ref b) => self.conflicts_buffer_all(b),
            &Key::Image(ref b) => self.conflicts_image_all(b),
            &Key::FramebufferAttachment(ref b, idx) => {
                self.conflicts_image_all(b.attachments()[idx as usize].parent())
            },
        }
    }
}

impl Eq for Key {
}

impl Hash for Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            &Key::Buffer(ref buf) => buf.conflict_key_all().hash(state),
            &Key::Image(ref img) => img.conflict_key_all().hash(state),
            &Key::FramebufferAttachment(ref fb, idx) => {
                let img = fb.attachments()[idx as usize].parent();
                img.conflict_key_all().hash(state)
            },
        }
    }
}

struct ResourceEntry {
    final_stages: PipelineStages,
    final_access: AccessFlagBits,
    exclusive: bool,
    initial_layout: Layout,
    final_layout: Layout,
}

impl<I> SubmitSyncBuilderLayer<I> {
    /// Builds a new layer that wraps around an existing builder.
    #[inline]
    pub fn new(inner: I) -> SubmitSyncBuilderLayer<I> {
        SubmitSyncBuilderLayer {
            inner: inner,
            resources: FnvHashMap::default(),
        }
    }

    // Adds a buffer to the list.
    fn add_buffer<B>(&mut self, buffer: &B, exclusive: bool)
        where B: BufferAccess + Send + Sync + Clone + 'static
    {
        // TODO: don't create the key every time ; https://github.com/rust-lang/rfcs/pull/1769
        let key = Key::Buffer(Box::new(buffer.clone()));
        match self.resources.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(ResourceEntry {
                    final_stages: PipelineStages { all_commands: true, ..PipelineStages::none() },     // FIXME:
                    final_access: AccessFlagBits::all(),        // FIXME:
                    exclusive: exclusive,
                    initial_layout: Layout::Undefined,
                    final_layout: Layout::Undefined,
                });
            },

            Entry::Occupied(mut entry) => {
                let entry = entry.get_mut();
                // TODO: update stages and access
                entry.exclusive = entry.exclusive || exclusive;
                entry.final_layout = Layout::Undefined;
            },
        }
    }

    // Adds an image to the list.
    fn add_image<T>(&mut self, image: &T, exclusive: bool)
        where T: ImageAccess + Send + Sync + Clone + 'static
    {
        let key = Key::Image(Box::new(image.clone()));
        match self.resources.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(ResourceEntry {
                    final_stages: PipelineStages { all_commands: true, ..PipelineStages::none() },     // FIXME:
                    final_access: AccessFlagBits::all(),        // FIXME:
                    exclusive: exclusive,
                    initial_layout: image.initial_layout_requirement(),     // FIXME:
                    final_layout: image.final_layout_requirement(),         // FIXME:
                });
            },

            Entry::Occupied(mut entry) => {
                let entry = entry.get_mut();
                // TODO: update stages and access
                entry.exclusive = entry.exclusive || exclusive;
                entry.final_layout = image.final_layout_requirement();         // FIXME:
            },
        }
    }

    // Adds a framebuffer to the list.
    fn add_framebuffer<F>(&mut self, framebuffer: &F)
        where F: FramebufferAbstract + Send + Sync + Clone + 'static
    {
        for index in 0 .. FramebufferAbstract::attachments(framebuffer).len() {
            let key = Key::FramebufferAttachment(Box::new(framebuffer.clone()), index as u32);
            let desc = framebuffer.attachment(index).expect("Wrong implementation of FramebufferAbstract trait");
            let final_layout = desc.final_layout;

            match self.resources.entry(key) {
                Entry::Vacant(entry) => {
                    entry.insert(ResourceEntry {
                        final_stages: PipelineStages { all_commands: true, ..PipelineStages::none() },     // FIXME:
                        final_access: AccessFlagBits::all(),        // FIXME:
                        exclusive: true,            // FIXME:
                        initial_layout: desc.initial_layout,
                        final_layout: final_layout,
                    });
                },

                Entry::Occupied(mut entry) => {
                    let entry = entry.get_mut();
                    // TODO: update stages and access
                    entry.exclusive = true;         // FIXME:
                    entry.final_layout = final_layout;
                },
            }
        }
    }
}

unsafe impl<I, O, E> CommandBufferBuild for SubmitSyncBuilderLayer<I>
    where I: CommandBufferBuild<Out = O, Err = E>
{
    type Out = SubmitSyncLayer<O>;
    type Err = E;

    #[inline]
    fn build(self) -> Result<Self::Out, E> {
        Ok(SubmitSyncLayer {
            inner: try!(self.inner.build()),
            resources: self.resources,
        })
    }
}

unsafe impl<I> DeviceOwned for SubmitSyncBuilderLayer<I>
    where I: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I> CommandBufferBuilder for SubmitSyncBuilderLayer<I>
    where I: CommandBufferBuilder
{
    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.inner.queue_family()
    }
}

// FIXME: implement manually
macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for SubmitSyncBuilderLayer<I>
            where I: AddCommand<$cmd, Out = O>
        {
            type Out = SubmitSyncBuilderLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Result<Self::Out, CommandAddError> {
                Ok(SubmitSyncBuilderLayer {
                    inner: AddCommand::add(self.inner, command)?,
                    resources: self.resources,
                })
            }
        }
    }
}

// FIXME: implement manually
pass_through!((S, Pl), commands_raw::CmdBindDescriptorSets<S, Pl>);
pass_through!((V), commands_raw::CmdBindVertexBuffers<V>);
pass_through!((C), commands_raw::CmdExecuteCommands<C>);

unsafe impl<I, O, Rp, F> AddCommand<commands_raw::CmdBeginRenderPass<Rp, F>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdBeginRenderPass<Rp, F>, Out = O>,
          F: FramebufferAbstract + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdBeginRenderPass<Rp, F>) -> Result<Self::Out, CommandAddError> {
        self.add_framebuffer(command.framebuffer());

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, B> AddCommand<commands_raw::CmdBindIndexBuffer<B>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdBindIndexBuffer<B>, Out = O>,
          B: BufferAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdBindIndexBuffer<B>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.buffer(), false);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, P> AddCommand<commands_raw::CmdBindPipeline<P>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdBindPipeline<P>, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdBindPipeline<P>) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, S, D> AddCommand<commands_raw::CmdBlitImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdBlitImage<S, D>, Out = O>,
          S: ImageAccess + Send + Sync + Clone + 'static,
          D: ImageAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdBlitImage<S, D>) -> Result<Self::Out, CommandAddError> {
        self.add_image(command.source(), false);
        self.add_image(command.destination(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdClearAttachments> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdClearAttachments, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdClearAttachments) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, S, D> AddCommand<commands_raw::CmdCopyBuffer<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdCopyBuffer<S, D>, Out = O>,
          S: BufferAccess + Send + Sync + Clone + 'static,
          D: BufferAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdCopyBuffer<S, D>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.source(), false);
        self.add_buffer(command.destination(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, S, D> AddCommand<commands_raw::CmdCopyBufferToImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdCopyBufferToImage<S, D>, Out = O>,
          S: BufferAccess + Send + Sync + Clone + 'static,
          D: ImageAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdCopyBufferToImage<S, D>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.source(), false);
        self.add_image(command.destination(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, S, D> AddCommand<commands_raw::CmdCopyImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdCopyImage<S, D>, Out = O>,
          S: ImageAccess + Send + Sync + Clone + 'static,
          D: ImageAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdCopyImage<S, D>) -> Result<Self::Out, CommandAddError> {
        self.add_image(command.source(), false);
        self.add_image(command.destination(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdDispatchRaw> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdDispatchRaw, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdDispatchRaw) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdDrawRaw> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdDrawRaw, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdDrawRaw) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdDrawIndexedRaw> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdDrawIndexedRaw, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdDrawIndexedRaw) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, B> AddCommand<commands_raw::CmdDrawIndirectRaw<B>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdDrawIndirectRaw<B>, Out = O>,
          B: BufferAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdDrawIndirectRaw<B>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.buffer(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdEndRenderPass> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdEndRenderPass, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdEndRenderPass) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, B> AddCommand<commands_raw::CmdFillBuffer<B>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdFillBuffer<B>, Out = O>,
          B: BufferAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdFillBuffer<B>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.buffer(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdNextSubpass> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdNextSubpass, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdNextSubpass) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, Pc, Pl> AddCommand<commands_raw::CmdPushConstants<Pc, Pl>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdPushConstants<Pc, Pl>, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdPushConstants<Pc, Pl>) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, S, D> AddCommand<commands_raw::CmdResolveImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdResolveImage<S, D>, Out = O>,
          S: ImageAccess + Send + Sync + Clone + 'static,
          D: ImageAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdResolveImage<S, D>) -> Result<Self::Out, CommandAddError> {
        self.add_image(command.source(), false);
        self.add_image(command.destination(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdSetEvent> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdSetEvent, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdSetEvent) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdSetState> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdSetState, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdSetState) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

unsafe impl<I, O, B, D> AddCommand<commands_raw::CmdUpdateBuffer<B, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdUpdateBuffer<B, D>, Out = O>,
          B: BufferAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdUpdateBuffer<B, D>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.buffer(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            resources: self.resources,
        })
    }
}

/// Layer around a command buffer that handles synchronization between command buffers.
pub struct SubmitSyncLayer<I> {
    inner: I,
    resources: FnvHashMap<Key, ResourceEntry>,
}

unsafe impl<I> CommandBuffer for SubmitSyncLayer<I> where I: CommandBuffer {
    type Pool = I::Pool;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<I::Pool> {
        self.inner.inner()
    }

    fn submit_check(&self, future: &GpuFuture, queue: &Queue) -> Result<(), Box<Error>> {
        for (key, entry) in self.resources.iter() {
            match key {
                &Key::Buffer(ref buf) => {
                    if future.check_buffer_access(&buf, entry.exclusive, queue).is_ok() {
                        unsafe { buf.increase_gpu_lock(); }
                        continue;
                    }

                    if !buf.try_gpu_lock(entry.exclusive, queue) {
                        panic!()    // FIXME: return Err();
                    }
                },

                &Key::Image(ref img) => {
                    if future.check_image_access(img, entry.initial_layout, entry.exclusive, queue).is_ok() {
                        unsafe { img.increase_gpu_lock(); }
                        continue;
                    }

                    if !img.try_gpu_lock(entry.exclusive, queue) {
                        panic!()    // FIXME: return Err();
                    }
                },

                &Key::FramebufferAttachment(ref fb, idx) => {
                    let img = fb.attachments()[idx as usize].parent();

                    if future.check_image_access(img, entry.initial_layout, entry.exclusive, queue).is_ok() {
                        unsafe { img.increase_gpu_lock(); }
                        continue;
                    }

                    if !img.try_gpu_lock(entry.exclusive, queue) {
                        panic!()    // FIXME: return Err();
                    }
                },
            }
        }

        // FIXME: pipeline barriers if necessary?

        Ok(())
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        // TODO: check the queue family

        // We can't call `.get()` on the HashMap because of the `Borrow` requirement that's
        // unimplementable on our key type.
        // TODO:

        for (key, value) in self.resources.iter() {
            if !key.conflicts_buffer_all(buffer) {
                continue;
            }

            if !value.exclusive && exclusive {
                return Err(AccessCheckError::Denied(AccessError::ExclusiveDenied));
            }

            return Ok(Some((value.final_stages, value.final_access)));
        }

        Err(AccessCheckError::Unknown)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: Layout, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        // TODO: check the queue family

        // We can't call `.get()` on the HashMap because of the `Borrow` requirement that's
        // unimplementable on our key type.
        // TODO:

        for (key, value) in self.resources.iter() {
            if !key.conflicts_image_all(image) {
                continue;
            }

            if value.final_layout != layout {
                return Err(AccessCheckError::Denied(AccessError::UnexpectedImageLayout {
                    allowed: value.final_layout,
                    requested: layout,
                }));
            }

            if !value.exclusive && exclusive {
                return Err(AccessCheckError::Denied(AccessError::ExclusiveDenied));
            }

            return Ok(Some((value.final_stages, value.final_access)));
        }

        Err(AccessCheckError::Unknown)
    }
}

unsafe impl<I> DeviceOwned for SubmitSyncLayer<I> where I: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}
