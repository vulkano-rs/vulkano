// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferViewRef;
use buffer::TrackedBuffer;
use command_buffer::cmd::CommandsListSink;
use command_buffer::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::TrackedDescriptorSet;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::sys::UnsafeDescriptorSet;
use descriptor::descriptor_set::sys::DescriptorWrite;
use descriptor::pipeline_layout::PipelineLayoutRef;
use device::Queue;
use image::TrackedImage;
use image::TrackedImageView;
use image::sys::Layout;
use sync::AccessFlagBits;
use sync::Fence;
use sync::PipelineStages;

/// A simple immutable descriptor set.
///
/// It is named "simple" because creating such a descriptor set allocates from a pool, and because
/// it can't be modified once created. It is sufficient for most usages, but in some situations
/// you may wish to use something more optimized instead.
///
/// In order to build a `SimpleDescriptorSet`, you need to use a `SimpleDescriptorSetBuilder`. But
/// the easiest way is to use the `simple_descriptor_set!` macro.
///
/// The template parameter of the `SimpleDescriptorSet` is very complex, and you shouldn't try to
/// express it explicitely. If you want to store your descriptor set in a struct or in a `Vec` for
/// example, you are encouraged to turn `SimpleDescriptorSet` into a `Box<DescriptorSet>` or a
/// `Arc<DescriptorSet>`.
///
/// # Example
// TODO:
pub struct SimpleDescriptorSet<R> {
    inner: UnsafeDescriptorSet,
    resources: R,
}

impl<R> SimpleDescriptorSet<R> {
    /// Returns the layout used to create this descriptor set.
    #[inline]
    pub fn set_layout(&self) -> &Arc<UnsafeDescriptorSetLayout> {
        self.inner.layout()
    }
}

unsafe impl<R> DescriptorSet for SimpleDescriptorSet<R> {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        &self.inner
    }
}

unsafe impl<R, S> TrackedDescriptorSet<S> for SimpleDescriptorSet<R>
    where R: SimpleDescriptorSetResourcesCollection<S>
{
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        self.resources.add_transition(sink)
    }

    #[inline]
    unsafe fn transition(&self, states: &mut S, num_command: usize)
                         -> (usize, PipelineBarrierBuilder)
    {
        self.resources.transition(states, num_command)
    }

    #[inline]
    unsafe fn finish(&self, i: &mut S, o: &mut S) -> PipelineBarrierBuilder {
        self.resources.finish(i, o)
    }

    #[inline]
    unsafe fn on_submit<F>(&self, states: &S, queue: &Arc<Queue>, fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        self.resources.on_submit(states, queue, fence)
    }
}

/// Builds a descriptor set in the form of a `SimpleDescriptorSet` object.
// TODO: more doc
#[macro_export]
macro_rules! simple_descriptor_set {
    ($layout:expr, $set_num:expr, {$($name:ident: $val:expr),*$(,)*}) => ({
        use $crate::descriptor::descriptor_set::SimpleDescriptorSetBuilder;
        use $crate::descriptor::descriptor_set::SimpleDescriptorSetBufferExt;

        let builder = SimpleDescriptorSetBuilder::new($layout, $set_num);

        $(
            let builder = $val.add_me(builder, stringify!($name));
        )*

        builder.build()
    });
}

/// Prototype of a `SimpleDescriptorSet`.
///
/// > **Note**: You are encouraged to use the `simple_descriptor_set!` macro instead of
/// > manipulating these internals.
///
/// # Example
// TODO: example here
pub struct SimpleDescriptorSetBuilder<L, R> {
    layout: L,
    set_id: usize,
    writes: Vec<DescriptorWrite>,
    resources: R,
}

impl<L> SimpleDescriptorSetBuilder<L, ()> where L: PipelineLayoutRef {
    /// Builds a new prototype for a `SimpleDescriptorSet`. Requires a reference to a pipeline
    /// layout, and the id of the set within the layout.
    ///
    /// # Panic
    ///
    /// - Panics if the set id is out of range.
    ///
    pub fn new(layout: L, set_id: usize) -> SimpleDescriptorSetBuilder<L, ()> {
        assert!(layout.desc().num_sets() > set_id);

        SimpleDescriptorSetBuilder {
            layout: layout,
            set_id: set_id,
            writes: Vec::new(),
            resources: (),
        }
    }
}

impl<L, R> SimpleDescriptorSetBuilder<L, R> where L: PipelineLayoutRef {
    /// Builds a `SimpleDescriptorSet` from the builder.
    pub fn build(self) -> SimpleDescriptorSet<R> {
        // TODO: check that we filled everything
        // TODO: don't create a pool every time
        let pool = Arc::new(DescriptorPool::raw(self.layout.device()).unwrap());       // FIXME: error
        let set_layout = self.layout.descriptor_set_layout(self.set_id).unwrap();       // FIXME: error

        let set = unsafe {
            let mut set = UnsafeDescriptorSet::uninitialized_raw(&pool, set_layout).unwrap();      // FIXME: error
            set.write(self.writes.into_iter());
            set
        };

        SimpleDescriptorSet {
            inner: set,
            resources: self.resources,
        }
    }
}

/// Trait implemented on buffer values that can be appended to a simple descriptor set builder.
pub unsafe trait SimpleDescriptorSetBufferExt<L, R> {
    type Out;

    // TODO: return Result
    fn add_me(self, i: SimpleDescriptorSetBuilder<L, R>, name: &str)
              -> SimpleDescriptorSetBuilder<L, Self::Out>;
}

unsafe impl<L, R, T> SimpleDescriptorSetBufferExt<L, R> for T
    where T: TrackedBuffer, L: PipelineLayoutRef
{
    type Out = (R, SimpleDescriptorSetBuf<T>);

    fn add_me(self, i: SimpleDescriptorSetBuilder<L, R>, name: &str)
              -> SimpleDescriptorSetBuilder<L, Self::Out>
    {
        let (set_id, binding_id) = i.layout.desc().descriptor_by_name(name).unwrap();    // TODO: Result instead
        assert_eq!(set_id, i.set_id);       // TODO: Result instead
        let desc = i.layout.desc().descriptor(set_id, binding_id).unwrap();     // TODO: Result instead

        //i.writes.push(DescriptorWrite:);

        SimpleDescriptorSetBuilder {
            layout: i.layout,
            set_id: i.set_id,
            writes: i.writes,
            resources: (i.resources, SimpleDescriptorSetBuf {
                buffer: self,
                write: !desc.readonly,
                stage: PipelineStages::none(),      // FIXME:
                access: AccessFlagBits::none(),     // FIXME:
            })
        }
    }
}

/// Internal trait related to the `SimpleDescriptorSet` system.
pub unsafe trait SimpleDescriptorSetResourcesCollection<States> {
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>);

    /// Extracts the states relevant to the buffers and images contained in the descriptor set.
    /// Then transitions them to the right state.
    // TODO: must return a Result if multiple elements conflict with one another
    unsafe fn transition(&self, states: &mut States, num_command: usize)
                         -> (usize, PipelineBarrierBuilder);

    unsafe fn finish(&self, in_s: &mut States, out: &mut States) -> PipelineBarrierBuilder;

    // TODO: write docs
    unsafe fn on_submit<F>(&self, &States, queue: &Arc<Queue>, fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>;
}

unsafe impl<S> SimpleDescriptorSetResourcesCollection<S> for () {
    #[inline]
    fn add_transition<'a>(&'a self, _: &mut CommandsListSink<'a>) {
    }

    #[inline]
    unsafe fn transition(&self, _: &mut S, _: usize) -> (usize, PipelineBarrierBuilder) {
        (0, PipelineBarrierBuilder::new())
    }

    #[inline]
    unsafe fn finish(&self, _: &mut S, _: &mut S) -> PipelineBarrierBuilder {
        PipelineBarrierBuilder::new()
    }

    #[inline]
    unsafe fn on_submit<F>(&self, _: &S, queue: &Arc<Queue>, fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        SubmitInfo::empty()
    }
}

/// Internal object related to the `SimpleDescriptorSet` system.
pub struct SimpleDescriptorSetBuf<B> {
    buffer: B,
    write: bool,
    stage: PipelineStages,
    access: AccessFlagBits,
}

unsafe impl<B, S> SimpleDescriptorSetResourcesCollection<S> for SimpleDescriptorSetBuf<B>
    where B: TrackedBuffer<S>
{
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        // TODO: sink.add_buffer_transition(&self.buffer, 0, self.buffer.size(), self.write);
        unimplemented!()
    }

    #[inline]
    unsafe fn transition(&self, states: &mut S, num_command: usize)
                         -> (usize, PipelineBarrierBuilder)
    {
        let trans = self.buffer.transition(states, num_command, 0, self.buffer.size(),
                                           self.write, self.stage, self.access);
        
        if let Some(trans) = trans {
            let n = trans.after_command_num;
            let mut b = PipelineBarrierBuilder::new();
            b.add_buffer_barrier_request(&self.buffer, trans);
            (n, b)
        } else {
            (0, PipelineBarrierBuilder::new())
        }
    }

    #[inline]
    unsafe fn finish(&self, in_s: &mut S, out: &mut S) -> PipelineBarrierBuilder {
        if let Some(trans) = self.buffer.finish(in_s, out) {
            let mut b = PipelineBarrierBuilder::new();
            b.add_buffer_barrier_request(&self.buffer, trans);
            b
        } else {
            PipelineBarrierBuilder::new()
        }
    }

    unsafe fn on_submit<F>(&self, _: &S, queue: &Arc<Queue>, fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        unimplemented!()        // FIXME:
    }
}

/// Internal object related to the `SimpleDescriptorSet` system.
pub struct SimpleDescriptorSetBufView<V> where V: BufferViewRef {
    view: V,
    write: bool,
    stage: PipelineStages,
    access: AccessFlagBits,
}

unsafe impl<V, S> SimpleDescriptorSetResourcesCollection<S> for SimpleDescriptorSetBufView<V>
    where V: BufferViewRef, V::Buffer: TrackedBuffer<S>
{
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        // TODO: sink.add_buffer_transition(&self.view.buffer(), 0, self.view.buffer().size(), self.write);
        unimplemented!()
    }

    #[inline]
    unsafe fn transition(&self, states: &mut S, num_command: usize)
                         -> (usize, PipelineBarrierBuilder)
    {
        let trans = self.view.view().buffer()
                        .transition(states, num_command, 0, self.view.view().buffer().size(),
                                    self.write, self.stage, self.access);
        
        if let Some(trans) = trans {
            let n = trans.after_command_num;
            let mut b = PipelineBarrierBuilder::new();
            b.add_buffer_barrier_request(&self.view.view().buffer(), trans);
            (n, b)
        } else {
            (0, PipelineBarrierBuilder::new())
        }
    }

    #[inline]
    unsafe fn finish(&self, in_s: &mut S, out: &mut S) -> PipelineBarrierBuilder {
        if let Some(trans) = self.view.view().buffer().finish(in_s, out) {
            let mut b = PipelineBarrierBuilder::new();
            b.add_buffer_barrier_request(&self.view.view().buffer(), trans);
            b
        } else {
            PipelineBarrierBuilder::new()
        }
    }

    unsafe fn on_submit<Fe>(&self, _: &S, queue: &Arc<Queue>, fence: Fe) -> SubmitInfo
        where Fe: FnMut() -> Arc<Fence>
    {
        unimplemented!()        // FIXME:
    }
}

/// Internal object related to the `SimpleDescriptorSet` system.
pub struct SimpleDescriptorSetImg<I> {
    image: I,
    write: bool,
    first_mipmap: u32,
    num_mipmaps: u32,
    first_layer: u32,
    num_layers: u32,
    layout: Layout,
    stage: PipelineStages,
    access: AccessFlagBits,
}

unsafe impl<I, S> SimpleDescriptorSetResourcesCollection<S> for SimpleDescriptorSetImg<I>
    where I: TrackedImageView<S>
{
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        // TODO: sink.add_image_transition(&self.image, self.first_layer, self.num_layers,
        //                                 self.first_mipmap, self.num_mipmaps, self.write,
        //                                 self.layout);
        unimplemented!()
    }

    #[inline]
    unsafe fn transition(&self, states: &mut S, num_command: usize)
                         -> (usize, PipelineBarrierBuilder)
    {
        // TODO: check whether mipmaps and layers are in range

        let trans = self.image.image()
                        .transition(states, num_command, self.first_mipmap, self.num_mipmaps,
                                    self.first_layer, self.num_layers, self.write, self.layout,
                                    self.stage, self.access);

        if let Some(trans) = trans {
            let n = trans.after_command_num;
            let mut b = PipelineBarrierBuilder::new();
            b.add_image_barrier_request(&self.image.image(), trans);
            (n, b)
        } else {
            (0, PipelineBarrierBuilder::new())
        }
    }

    #[inline]
    unsafe fn finish(&self, in_s: &mut S, out: &mut S) -> PipelineBarrierBuilder {
        if let Some(trans) = self.image.image().finish(in_s, out) {
            let mut b = PipelineBarrierBuilder::new();
            b.add_image_barrier_request(&self.image.image(), trans);
            b
        } else {
            PipelineBarrierBuilder::new()
        }
    }

    unsafe fn on_submit<Fe>(&self, _: &S, queue: &Arc<Queue>, fence: Fe) -> SubmitInfo
        where Fe: FnMut() -> Arc<Fence>
    {
        unimplemented!()        // FIXME:
    }
}

unsafe impl<S, A, B> SimpleDescriptorSetResourcesCollection<S> for (A, B)
    where A: SimpleDescriptorSetResourcesCollection<S>,
          B: SimpleDescriptorSetResourcesCollection<S>
{
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        self.0.add_transition(sink);
        self.1.add_transition(sink);
    }

    #[inline]
    unsafe fn transition(&self, states: &mut S, num_command: usize)
                            -> (usize, PipelineBarrierBuilder)
    {
        let (mut nc, mut barrier) = self.0.transition(states, num_command);
        let (n, b) = self.1.transition(states, num_command);
        nc = cmp::max(nc, n);
        barrier.merge(b);
        debug_assert!(nc <= num_command);
        (nc, barrier)
    }

    #[inline]
    unsafe fn finish(&self, in_s: &mut S, out: &mut S) -> PipelineBarrierBuilder {
        let mut barrier = self.0.finish(in_s, out);
        barrier.merge(self.1.finish(in_s, out));
        barrier
    }

    unsafe fn on_submit<F>(&self, _: &S, queue: &Arc<Queue>, fence: F)
                        -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        unimplemented!()
    }
}
