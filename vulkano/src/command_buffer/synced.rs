// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::DerefMut;
use std::sync::Arc;
use fnv::FnvHashMap;
use smallvec::SmallVec;

use buffer::BufferAccess;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferExecError;
use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolAlloc;
use command_buffer::pool::CommandPoolBuilderAlloc;
use command_buffer::sys::Flags;
use command_buffer::sys::Kind;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use command_buffer::sys::UnsafeCommandBufferBuilderBindVertexBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilderBufferImageCopy;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use descriptor::descriptor::ShaderStages;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use format::ClearValue;
use framebuffer::Framebuffer;
use framebuffer::FramebufferAbstract;
use framebuffer::RenderPass;
use framebuffer::RenderPassAbstract;
use image::ImageLayout;
use image::ImageAccess;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use pipeline::input_assembly::IndexType;
use pipeline::vertex::VertexSource;
use pipeline::viewport::Scissor;
use pipeline::viewport::Viewport;
use sync::AccessCheckError;
use sync::AccessError;
use sync::AccessFlagBits;
use sync::Event;
use sync::PipelineStages;
use sync::GpuFuture;
use OomError;

pub struct SyncCommandBufferBuilder<P> {
    inner: UnsafeCommandBufferBuilder<P>,
    resources: FnvHashMap<Key<'static>, ResourceEntry>,
    // Contains the objects other than the ones in `resources` that must be kept alive while the
    // command buffer is being executed (eg. pipeline objects, ...).
    keep_alive: Vec<Box<KeepAlive + Send + Sync>>,
}

pub enum SyncCommandBufferBuilderError {

}

// Trait implemented on everything, so that we can turn any `T` into a `Box<KeepAlive>`.
trait KeepAlive {}
impl<T> KeepAlive for T {}

// Key that identifies a resource. Implements `PartialEq`, `Eq` and `Hash` so that two resources
// that conflict with each other compare equal.
enum Key<'a> {
    // A buffer.
    Buffer(Box<BufferAccess + Send + Sync>),
    // References to a buffer. This variant of the key must never be stored in a hashmap. Instead
    // it must be used only when creating a temporary key to lookup an entry in said hashmap.
    BufferRef(&'a BufferAccess),
    // An image.
    Image(Box<ImageAccess + Send + Sync>),
    // References to a buffer. This variant of the key must never be stored in a hashmap. Instead
    // it must be used only when creating a temporary key to lookup an entry in said hashmap.
    ImageRef(&'a ImageAccess),
    FramebufferAttachment(Box<FramebufferAbstract + Send + Sync>, u32),
}

// `BufferRef` and `ImageRef` don't implement `Send`/`Sync`, but all other variants do. Since these
// two exceptions must never be stored in a hashmap, we implement `Send`/`Sync` manually so that
// the hashmap implements `Send` and `Sync` as well.
unsafe impl<'a> Send for Key<'a> {}
unsafe impl<'a> Sync for Key<'a> {}

impl<'a> Key<'a> {
    #[inline]
    fn conflicts_buffer_all(&self, buf: &BufferAccess) -> bool {
        match self {
            &Key::Buffer(ref a) => a.conflicts_buffer_all(buf),
            &Key::BufferRef(ref a) => a.conflicts_buffer_all(buf),
            &Key::Image(ref a) => a.conflicts_buffer_all(buf),
            &Key::ImageRef(ref a) => a.conflicts_buffer_all(buf),
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
            &Key::BufferRef(ref a) => a.conflicts_image_all(img),
            &Key::Image(ref a) => a.conflicts_image_all(img),
            &Key::ImageRef(ref a) => a.conflicts_image_all(img),
            &Key::FramebufferAttachment(ref b, idx) => {
                let b = b.attachments()[idx as usize].parent();
                b.conflicts_image_all(img)
            },
        }
    }
}

impl<'a> PartialEq for Key<'a> {
    #[inline]
    fn eq(&self, other: &Key) -> bool {
        match other {
            &Key::Buffer(ref b) => self.conflicts_buffer_all(b),
            &Key::BufferRef(ref b) => self.conflicts_buffer_all(b),
            &Key::Image(ref b) => self.conflicts_image_all(b),
            &Key::ImageRef(ref b) => self.conflicts_image_all(b),
            &Key::FramebufferAttachment(ref b, idx) => {
                self.conflicts_image_all(b.attachments()[idx as usize].parent())
            },
        }
    }
}

impl<'a> Eq for Key<'a> {
}

impl<'a> Hash for Key<'a> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            &Key::Buffer(ref buf) => buf.conflict_key_all().hash(state),
            &Key::BufferRef(ref buf) => buf.conflict_key_all().hash(state),
            &Key::Image(ref img) => img.conflict_key_all().hash(state),
            &Key::ImageRef(ref img) => img.conflict_key_all().hash(state),
            &Key::FramebufferAttachment(ref fb, idx) => {
                let img = fb.attachments()[idx as usize].parent();
                img.conflict_key_all().hash(state)
            },
        }
    }
}

// Synchronization state of a resource.
struct ResourceEntry {
    final_stages: PipelineStages,
    final_access: AccessFlagBits,
    exclusive: bool,
    initial_layout: ImageLayout,
    final_layout: ImageLayout,
}

// TODO: should check conflicts within each command
struct Binder<'r> {
    resources: &'r mut FnvHashMap<Key<'static>, ResourceEntry>,
    insertions: SmallVec<[Option<ResourceEntry>; 16]>,
}

fn start<'r>(resources: &'r mut FnvHashMap<Key<'static>, ResourceEntry>) -> Binder<'r> {
    Binder {
        resources,
        insertions: SmallVec::new(),
    }
}

impl<'r> Binder<'r> {
    fn add_buffer<B>(&mut self, buffer: &B, exclusive: bool, stages: PipelineStages,
                     access: AccessFlagBits)
        where B: BufferAccess
    {
        // TODO: yay, can't even call `get_mut` on the hash map without a Key that is 'static
        //       itself ; Rust needs HKTs for that
        let key: Key = Key::BufferRef(buffer);
        let key: Key<'static> = unsafe { mem::transmute(key) };

        match self.resources.get_mut(&key) {
            Some(entry) => {
                // TODO: remove some stages and accesses when there's an "overflow"?
                entry.final_stages = entry.final_stages | stages;
                entry.final_access = entry.final_access | access;
                entry.exclusive = entry.exclusive || exclusive;
                entry.final_layout = ImageLayout::Undefined;
                self.insertions.push(None);
            },
            None => {
                self.insertions.push(Some(ResourceEntry {
                    final_stages: stages,
                    final_access: access,
                    exclusive: exclusive,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::Undefined,
                }));
            },
        }
    }

    fn add_image<I>(&mut self, image: &I, exclusive: bool, stages: PipelineStages,
                    access: AccessFlagBits, initial_layout: ImageLayout, final_layout: ImageLayout)
        where I: ImageAccess
    {
        // TODO: yay, can't even call `get_mut` on the hash map without a Key that is 'static
        //       itself ; Rust needs HKTs for that
        let key: Key = Key::ImageRef(image);
        let key: Key<'static> = unsafe { mem::transmute(key) };

        match self.resources.get_mut(&key) {
            Some(entry) => {
                // TODO: exclusive accss if transition required?
                entry.exclusive = entry.exclusive || exclusive;
                // TODO: remove some stages and accesses when there's an "overflow"?
                entry.final_stages = entry.final_stages | stages;
                entry.final_access = entry.final_access | access;
                entry.final_layout = final_layout;
                self.insertions.push(None);
            },
            None => {
                self.insertions.push(Some(ResourceEntry {
                    final_stages: stages,
                    final_access: access,
                    exclusive: exclusive,
                    initial_layout: initial_layout,
                    final_layout: final_layout,
                }));
            }
        }
    }

    fn flush_pipeline_barrier<P>(&mut self, inner: &mut UnsafeCommandBufferBuilder<P>) {
        // TODO:
    }

    fn finish_buf<B>(&mut self, buffer: B)
        where B: BufferAccess + Send + Sync + 'static
    {
        match self.insertions.remove(0) {
            None => (),
            Some(entry) => {
                let prev_value = self.resources.insert(Key::Buffer(Box::new(buffer) as Box<_>), entry);
                debug_assert!(prev_value.is_none());
            },
        }
    }

    fn finish_img<I>(&mut self, image: I)
        where I: ImageAccess + Send + Sync + 'static
    {
        match self.insertions.remove(0) {
            None => (),
            Some(entry) => {
                let prev_value = self.resources.insert(Key::Image(Box::new(image) as Box<_>), entry);
                debug_assert!(prev_value.is_none());
            },
        }
    }
}

impl<P> SyncCommandBufferBuilder<P> {
    pub unsafe fn new<Pool, R, F, A>(pool: &Pool, kind: Kind<R, F>, flags: Flags)
                                     -> Result<SyncCommandBufferBuilder<P>, OomError>
        where Pool: CommandPool<Builder = P, Alloc = A>,
              P: CommandPoolBuilderAlloc<Alloc = A>,
              A: CommandPoolAlloc,
              R: RenderPassAbstract,
              F: FramebufferAbstract
    {
        let cmd = UnsafeCommandBufferBuilder::new(pool, kind, flags)?;
        Ok(SyncCommandBufferBuilder::from_unsafe_cmd(cmd))
    }

    #[inline]
    fn from_unsafe_cmd(cmd: UnsafeCommandBufferBuilder<P>) -> SyncCommandBufferBuilder<P> {
        SyncCommandBufferBuilder {
            inner: cmd,
            resources: FnvHashMap::default(),
            keep_alive: Vec::new(),
        }
    }

    /// Builds the command buffer.
    #[inline]
    pub fn build(mut self) -> Result<SyncCommandBuffer<P::Alloc>, OomError>
        where P: CommandPoolBuilderAlloc
    {
        // TODO: only do this if we don't have the one time submit flag
        self.resources.shrink_to_fit();
        self.keep_alive.shrink_to_fit();

        Ok(SyncCommandBuffer {
            inner: self.inner.build()?,
            resources: self.resources,
            keep_alive: self.keep_alive,
        })
    }

    // Adds a framebuffer to the list.
    fn add_framebuffer<F>(&mut self, framebuffer: F)
        where F: FramebufferAbstract + Send + Sync + 'static
    {
        /*// TODO: slow
        for index in 0 .. FramebufferAbstract::attachments(framebuffer).len() {
            let key = Key::FramebufferAttachment(Box::new(framebuffer.clone()), index as u32);
            let desc = framebuffer.attachment_desc(index).expect("Wrong implementation of FramebufferAbstract trait");
            let image = FramebufferAbstract::attachments(framebuffer)[index];

            let initial_layout = {
                match desc.initial_layout {
                    ImageLayout::Undefined | ImageLayout::Preinitialized => desc.initial_layout,
                    _ => image.parent().initial_layout_requirement(),
                }
            };

            let final_layout = {
                match desc.final_layout {
                    ImageLayout::Undefined | ImageLayout::Preinitialized => desc.final_layout,
                    _ => image.parent().final_layout_requirement(),
                }
            };

            match self.resources.entry(key) {
                Entry::Vacant(entry) => {
                    entry.insert(ResourceEntry {
                        final_stages: PipelineStages { all_commands: true, ..PipelineStages::none() },     // FIXME:
                        final_access: AccessFlagBits::all(),        // FIXME:
                        exclusive: true,            // FIXME:
                        initial_layout: initial_layout,
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
        }*/
    }

    /// Calls `vkBeginRenderPass` on the builder.
    #[inline]
    pub unsafe fn begin_render_pass<F, I>(&mut self, framebuffer: F, secondary: bool,
                                          clear_values: I)
        where F: FramebufferAbstract + Send + Sync + 'static,
              I: Iterator<Item = ClearValue>
    {
        self.inner.begin_render_pass(&framebuffer, secondary, clear_values);
        self.add_framebuffer(framebuffer);
    }

    /// Calls `vkCmdBindIndexBuffer` on the builder.
    #[inline]
    pub unsafe fn bind_index_buffer<B>(&mut self, buffer: B, index_ty: IndexType)
        where B: BufferAccess + Send + Sync + 'static
    {
        let mut binder = start(&mut self.resources);
        binder.add_buffer(&buffer, false,
                          PipelineStages { vertex_input: true, .. PipelineStages::none() },
                          AccessFlagBits { index_read: true, .. AccessFlagBits::none() });
        binder.flush_pipeline_barrier(&mut self.inner);

        self.inner.bind_index_buffer(&buffer, index_ty);

        binder.finish_buf(buffer);
    }
    
    /// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_graphics<Gp>(&mut self, pipeline: Gp)
        where Gp: GraphicsPipelineAbstract + Send + Sync + 'static
    {
        self.inner.bind_pipeline_graphics(&pipeline);
        self.keep_alive.push(Box::new(pipeline) as Box<_>);
    }

    /// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_compute<Cp>(&mut self, pipeline: Cp)
        where Cp: ComputePipelineAbstract + Send + Sync + 'static
    {
        self.inner.bind_pipeline_compute(&pipeline);
        self.keep_alive.push(Box::new(pipeline) as Box<_>);
    }

    #[inline]
    pub fn bind_descriptor_sets(&mut self) -> SyncCommandBufferBuilderBindDescriptorSets<P> {
        SyncCommandBufferBuilderBindDescriptorSets {
            builder: self,
            inner: SmallVec::new(),
        }
    }

    #[inline]
    pub fn bind_vertex_buffers(&mut self) -> SyncCommandBufferBuilderBindVertexBuffer<P> {
        SyncCommandBufferBuilderBindVertexBuffer {
            builder: self,
            inner: UnsafeCommandBufferBuilderBindVertexBuffer::new(),
        }
    }

    /// Calls `vkCmdCopyBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer<S, D, R>(&mut self, source: S, destination: D, regions: R)
        where S: BufferAccess + Send + Sync + 'static,
              D: BufferAccess + Send + Sync + 'static,
              R: Iterator<Item = (usize, usize, usize)>
    {
        let mut binder = start(&mut self.resources);
        binder.add_buffer(&source, false,
                          PipelineStages { transfer: true, .. PipelineStages::none() },
                          AccessFlagBits { transfer_read: true, .. AccessFlagBits::none() });
        binder.add_buffer(&destination, true,
                          PipelineStages { transfer: true, .. PipelineStages::none() },
                          AccessFlagBits { transfer_write: true, .. AccessFlagBits::none() });
        binder.flush_pipeline_barrier(&mut self.inner);

        self.inner.copy_buffer(&source, &destination, regions);

        binder.finish_buf(source);
        binder.finish_buf(destination);
    }

    /// Calls `vkCmdCopyBufferToImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer_to_image<S, D, R>(&mut self, source: S, destination: D,
                                                dest_layout: ImageLayout, regions: R)
        where S: BufferAccess + Send + Sync + 'static,
              D: ImageAccess + Send + Sync + 'static,
              R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>
    {
        let mut binder = start(&mut self.resources);
        binder.add_buffer(&source, false,
                          PipelineStages { transfer: true, .. PipelineStages::none() },
                          AccessFlagBits { transfer_read: true, .. AccessFlagBits::none() });
        binder.add_image(&destination, true,
                         PipelineStages { transfer: true, .. PipelineStages::none() },
                         AccessFlagBits { transfer_write: true, .. AccessFlagBits::none() },
                         dest_layout, dest_layout);
        binder.flush_pipeline_barrier(&mut self.inner);

        self.inner.copy_buffer_to_image(&source, &destination, dest_layout, regions);

        binder.finish_buf(source);
        binder.finish_img(destination);
    }

    /// Calls `vkCmdDispatch` on the builder.
    #[inline]
    pub unsafe fn dispatch(&mut self, dimensions: [u32; 3]) {
        self.inner.dispatch(dimensions);
    }

    /// Calls `vkCmdDispatchIndirect` on the builder.
    #[inline]
    pub unsafe fn dispatch_indirect<B>(&mut self, buffer: B)
        where B: BufferAccess + Send + Sync + 'static
    {
        let mut binder = start(&mut self.resources);
        binder.add_buffer(&buffer, false,
                          PipelineStages { draw_indirect: true, .. PipelineStages::none() },      // TODO: is draw_indirect correct?
                          AccessFlagBits { indirect_command_read: true, .. AccessFlagBits::none() });
        binder.flush_pipeline_barrier(&mut self.inner);

        self.inner.dispatch_indirect(&buffer);

        binder.finish_buf(buffer);
    }

    /// Calls `vkCmdDraw` on the builder.
    #[inline]
    pub unsafe fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32,
                       first_instance: u32)
    {
        self.inner.draw(vertex_count, instance_count, first_vertex, first_instance);
    }

    /// Calls `vkCmdDrawIndexed` on the builder.
    #[inline]
    pub unsafe fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32,
                               vertex_offset: i32, first_instance: u32)
    {
        self.inner.draw_indexed(index_count, instance_count, first_index, vertex_offset,
                                first_instance);
    }

    /// Calls `vkCmdDrawIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indirect<B>(&mut self, buffer: B, draw_count: u32, stride: u32)
        where B: BufferAccess + Send + Sync + 'static
    {
        let mut binder = start(&mut self.resources);
        binder.add_buffer(&buffer, false,
                          PipelineStages { draw_indirect: true, .. PipelineStages::none() },
                          AccessFlagBits { indirect_command_read: true, .. AccessFlagBits::none() });
        binder.flush_pipeline_barrier(&mut self.inner);

        self.inner.draw_indirect(&buffer, draw_count, stride);

        binder.finish_buf(buffer);
    }

    /// Calls `vkCmdDrawIndexedIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indexed_indirect<B>(&mut self, buffer: B, draw_count: u32, stride: u32)
        where B: BufferAccess + Send + Sync + 'static
    {
        let mut binder = start(&mut self.resources);
        binder.add_buffer(&buffer, false,
                          PipelineStages { draw_indirect: true, .. PipelineStages::none() },
                          AccessFlagBits { indirect_command_read: true, .. AccessFlagBits::none() });
        binder.flush_pipeline_barrier(&mut self.inner);

        self.inner.draw_indexed_indirect(&buffer, draw_count, stride);

        binder.finish_buf(buffer);
    }

    /// Calls `vkCmdEndRenderPass` on the builder.
    #[inline]
    pub unsafe fn end_render_pass(&mut self) {
        self.inner.end_render_pass();
    }

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer<B>(&mut self, buffer: B, data: u32)
        where B: BufferAccess + Send + Sync + 'static
    {
        let mut binder = start(&mut self.resources);
        binder.add_buffer(&buffer, true,
                          PipelineStages { transfer: true, .. PipelineStages::none() },
                          AccessFlagBits { transfer_write: true, .. AccessFlagBits::none() });
        binder.flush_pipeline_barrier(&mut self.inner);

        self.inner.fill_buffer(&buffer, data);

        binder.finish_buf(buffer);
    }

    /// Calls `vkCmdNextSubpass` on the builder.
    #[inline]
    pub unsafe fn next_subpass(&mut self, secondary: bool) {
        self.inner.next_subpass(secondary);
    }

    /// Calls `vkCmdPushConstants` on the builder.
    #[inline]
    pub unsafe fn push_constants<Pl, D>(&mut self, pipeline_layout: Pl, stages: ShaderStages,
                                        offset: u32, size: u32, data: &D)
        where Pl: PipelineLayoutAbstract + Send + Sync + 'static,
              D: ?Sized
    {
        self.inner.push_constants(&pipeline_layout, stages, offset, size, data);
        self.keep_alive.push(Box::new(pipeline_layout) as Box<_>);
    }

    /// Calls `vkCmdResetEvent` on the builder.
    #[inline]
    pub unsafe fn reset_event(&mut self, event: Arc<Event>, stages: PipelineStages) {
        self.inner.reset_event(&event, stages);
        self.keep_alive.push(Box::new(event) as Box<_>);
    }

    /// Calls `vkCmdSetBlendConstants` on the builder.
    #[inline]
    pub unsafe fn set_blend_constants(&mut self, constants: [f32; 4]) {
        self.inner.set_blend_constants(constants);
    }

    /// Calls `vkCmdSetDepthBias` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias(&mut self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        self.inner.set_depth_bias(constant_factor, clamp, slope_factor);
    }

    /// Calls `vkCmdSetDepthBounds` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds(&mut self, min: f32, max: f32) {
        self.inner.set_depth_bounds(min, max);
    }

    /// Calls `vkCmdSetEvent` on the builder.
    #[inline]
    pub unsafe fn set_event(&mut self, event: Arc<Event>, stages: PipelineStages) {
        self.inner.set_event(&event, stages);
        self.keep_alive.push(Box::new(event) as Box<_>);
    }

    /// Calls `vkCmdSetLineWidth` on the builder.
    #[inline]
    pub unsafe fn set_line_width(&mut self, line_width: f32) {
        self.inner.set_line_width(line_width);
    }

    // TODO: stencil states

    /// Calls `vkCmdSetScissor` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_scissor<I>(&mut self, first_scissor: u32, scissors: I)
        where I: Iterator<Item = Scissor>
    {
        self.inner.set_scissor(first_scissor, scissors);
    }

    /// Calls `vkCmdSetViewport` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_viewport<I>(&mut self, first_viewport: u32, viewports: I)
        where I: Iterator<Item = Viewport>
    {
        self.inner.set_viewport(first_viewport, viewports);
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    #[inline]
    pub unsafe fn update_buffer<B, D>(&mut self, buffer: B, data: &D)
        where B: BufferAccess + Send + Sync + 'static,
              D: ?Sized
    {
        let mut binder = start(&mut self.resources);
        binder.add_buffer(&buffer, true,
                          PipelineStages { transfer: true, .. PipelineStages::none() },
                          AccessFlagBits { transfer_write: true, .. AccessFlagBits::none() });
        binder.flush_pipeline_barrier(&mut self.inner);

        self.inner.update_buffer(&buffer, data);

        binder.finish_buf(buffer);
    }
}

unsafe impl<P> DeviceOwned for SyncCommandBufferBuilder<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

pub struct SyncCommandBufferBuilderBindDescriptorSets<'b, P: 'b> {
    builder: &'b mut SyncCommandBufferBuilder<P>,
    inner: SmallVec<[Box<DescriptorSet + Send + Sync>; 12]>,
}

impl<'b, P> SyncCommandBufferBuilderBindDescriptorSets<'b, P> {
    /// Adds a descriptor set to the list.
    #[inline]
    pub fn add<S>(&mut self, set: S)
        where S: DescriptorSet + Send + Sync + 'static
    {
        self.inner.push(Box::new(set));
    }

    #[inline]
    pub unsafe fn submit<Pl, I>(self, graphics: bool, pipeline_layout: Pl, first_binding: u32,
                                dynamic_offsets: I)
        where Pl: PipelineLayoutAbstract,
              I: Iterator<Item = u32>,
    {
        self.builder.inner.bind_descriptor_sets(graphics, &pipeline_layout, first_binding,
                                                self.inner.iter().map(|s| s.inner()),
                                                dynamic_offsets);
        
        for set in self.inner {
            self.builder.keep_alive.push(Box::new(set));
        }
    }
}

/// Prototype for a `vkCmdBindVertexBuffers`.
pub struct SyncCommandBufferBuilderBindVertexBuffer<'a, P: 'a> {
    builder: &'a mut SyncCommandBufferBuilder<P>,
    inner: UnsafeCommandBufferBuilderBindVertexBuffer,
}

impl<'a, P> SyncCommandBufferBuilderBindVertexBuffer<'a, P> {
    /// Adds a buffer to the list.
    #[inline]
    pub fn add<B>(&mut self, buffer: B)
        where B: BufferAccess + Send + Sync + 'static
    {
        self.inner.add(&buffer);
        // FIXME:
        /*self.builder.add_buffer(buffer, false,
                                PipelineStages { vertex_input: true, .. PipelineStages::none() },
                                AccessFlagBits { vertex_attribute_read: true, .. AccessFlagBits::none() });*/
    }

    #[inline]
    pub unsafe fn submit(self, first_binding: u32) {
        self.builder.inner.bind_vertex_buffers(first_binding, self.inner);
    }
}

pub struct SyncCommandBuffer<P> {
    inner: UnsafeCommandBuffer<P>,
    resources: FnvHashMap<Key<'static>, ResourceEntry>,
    // Contains the objects other than the ones in `resources` that must be kept alive while the
    // command buffer is being executed (eg. pipeline objects, ...).
    keep_alive: Vec<Box<KeepAlive + Send + Sync>>,
}

unsafe impl<P> CommandBuffer for SyncCommandBuffer<P> {
    type PoolAlloc = P;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::PoolAlloc> {
        &self.inner
    }

    fn prepare_submit(&self, future: &GpuFuture, queue: &Queue) -> Result<(), CommandBufferExecError> {
        // TODO: if at any point we return an error, we can't recover

        for (key, entry) in self.resources.iter() {
            match key {
                &Key::Buffer(ref buf) => {
                    let prev_err = match future.check_buffer_access(&buf, entry.exclusive, queue) {
                        Ok(_) => {
                            unsafe { buf.increase_gpu_lock(); }
                            continue;
                        },
                        Err(err) => err
                    };

                    match (buf.try_gpu_lock(entry.exclusive, queue), prev_err) {
                        (Ok(_), _) => (),
                        (Err(err), AccessCheckError::Unknown) => return Err(err.into()),
                        (_, AccessCheckError::Denied(err)) => return Err(err.into()),
                    }
                },

                &Key::Image(ref img) => {
                    let prev_err = match future.check_image_access(img, entry.initial_layout,
                                                                   entry.exclusive, queue)
                    {
                        Ok(_) => {
                            unsafe { img.increase_gpu_lock(); }
                            continue;
                        },
                        Err(err) => err
                    };

                    match (img.try_gpu_lock(entry.exclusive, queue), prev_err) {
                        (Ok(_), _) => (),
                        (Err(err), AccessCheckError::Unknown) => return Err(err.into()),
                        (_, AccessCheckError::Denied(err)) => return Err(err.into()),
                    }
                },

                &Key::FramebufferAttachment(ref fb, idx) => {
                    let img = fb.attachments()[idx as usize].parent();

                    let prev_err = match future.check_image_access(img, entry.initial_layout,
                                                                   entry.exclusive, queue)
                    {
                        Ok(_) => {
                            unsafe { img.increase_gpu_lock(); }
                            continue;
                        },
                        Err(err) => err
                    };

                    // FIXME: this is bad because dropping the submit sync layer doesn't drop the
                    //        attachments of the framebuffer, meaning that they will stay locked
                    match (img.try_gpu_lock(entry.exclusive, queue), prev_err) {
                        (Ok(_), _) => (),
                        (Err(err), AccessCheckError::Unknown) => return Err(err.into()),
                        (_, AccessCheckError::Denied(err)) => return Err(err.into()),
                    }
                },

                &Key::BufferRef(_) => unreachable!(),
                &Key::ImageRef(_) => unreachable!(),
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

        if let Some(value) = self.resources.get(&Key::BufferRef(buffer)) {
            if !value.exclusive && exclusive {
                return Err(AccessCheckError::Denied(AccessError::ExclusiveDenied));
            }

            return Ok(Some((value.final_stages, value.final_access)));
        }

        Err(AccessCheckError::Unknown)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        // TODO: check the queue family

        if let Some(value) = self.resources.get(&Key::ImageRef(image)) {
            if layout != ImageLayout::Undefined && value.final_layout != layout {
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

unsafe impl<P> DeviceOwned for SyncCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}
