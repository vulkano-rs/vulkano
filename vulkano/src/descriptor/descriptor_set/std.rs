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
use command_buffer::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::TrackedDescriptorSet;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::sys::UnsafeDescriptorSet;
use device::Queue;
use image::TrackedImage;
use image::TrackedImageView;
use image::sys::Layout;
use sync::AccessFlagBits;
use sync::Fence;
use sync::PipelineStages;

pub struct StdDescriptorSet<R> {
    inner: UnsafeDescriptorSet,
    resources: R,
}

impl<R> StdDescriptorSet<R> {
    ///
    /// # Safety
    ///
    /// - The resources must match the layout.
    ///
    pub unsafe fn new(pool: &Arc<DescriptorPool>, layout: &Arc<UnsafeDescriptorSetLayout>,
                      resources: R) -> StdDescriptorSet<R>
    {
        unimplemented!()
    }

    /// Returns the layout used to create this descriptor set.
    #[inline]
    pub fn layout(&self) -> &Arc<UnsafeDescriptorSetLayout> {
        self.inner.layout()
    }
}

unsafe impl<R> DescriptorSet for StdDescriptorSet<R> {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        &self.inner
    }
}

unsafe impl<R, S> TrackedDescriptorSet<S> for StdDescriptorSet<R>
    where R: StdDescriptorSetResourcesCollection<S>
{
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

// TODO: re-read docs
/// Collection of tracked resources. Makes it possible to treat multiple buffers and images as one.
pub unsafe trait StdDescriptorSetResourcesCollection<States> {
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

unsafe impl<S> StdDescriptorSetResourcesCollection<S> for () {
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

pub struct StdDescriptorSetBuf<B> {
    pub buffer: B,
    pub ty: StdDescriptorSetBufTy,
    pub write: bool,
    pub stage: PipelineStages,
    pub access: AccessFlagBits,
}

unsafe impl<B, S> StdDescriptorSetResourcesCollection<S> for StdDescriptorSetBuf<B>
    where B: TrackedBuffer<S>
{
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

pub enum StdDescriptorSetBufTy {
    StorageBuffer,
    UniformBuffer,
    DynamicStorageBuffer,
    DynamicUniformBuffer,
}

pub struct StdDescriptorSetBufView<V> where V: BufferViewRef {
    pub view: V,
    pub ty: StdDescriptorSetBufViewTy,
    pub write: bool,
    pub stage: PipelineStages,
    pub access: AccessFlagBits,
}

unsafe impl<V, S> StdDescriptorSetResourcesCollection<S> for StdDescriptorSetBufView<V>
    where V: BufferViewRef, V::Buffer: TrackedBuffer<S>
{
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

pub enum StdDescriptorSetBufViewTy {
    StorageBufferView,
    UniformBufferView,
}

pub struct StdDescriptorSetImg<I> {
    pub image: I,
    pub ty: StdDescriptorSetImgTy,
    pub write: bool,
    pub first_mipmap: u32,
    pub num_mipmaps: u32,
    pub first_layer: u32,
    pub num_layers: u32,
    pub layout: Layout,
    pub stage: PipelineStages,
    pub access: AccessFlagBits,
}

unsafe impl<I, S> StdDescriptorSetResourcesCollection<S> for StdDescriptorSetImg<I>
    where I: TrackedImageView<S>
{
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

pub enum StdDescriptorSetImgTy {
    StorageImage,
    SampledImage,
}

macro_rules! tuple_impl {
    ($first:ident, $($rest:ident),+) => (
        unsafe impl<S, $first, $($rest),+> StdDescriptorSetResourcesCollection<S> for ($first $(, $rest)+)
            where $first: StdDescriptorSetResourcesCollection<S>,
                  $($rest: StdDescriptorSetResourcesCollection<S>),+
        {
            #[inline]
            unsafe fn transition(&self, states: &mut S, num_command: usize)
                                 -> (usize, PipelineBarrierBuilder)
            {
                #![allow(non_snake_case)]
                let &(ref $first $(, ref $rest)+) = self;
                let (mut nc, mut barrier) = $first.transition(states, num_command);
                $({
                    let (n, b) = $rest.transition(states, num_command);
                    nc = cmp::max(nc, n);
                    barrier.merge(b);
                })+
                debug_assert!(nc <= num_command);
                (nc, barrier)
            }

            #[inline]
            unsafe fn finish(&self, in_s: &mut S, out: &mut S) -> PipelineBarrierBuilder {
                #![allow(non_snake_case)]
                let &(ref $first $(, ref $rest)+) = self;
                let mut barrier = $first.finish(in_s, out);
                $(
                    barrier.merge($rest.finish(in_s, out));
                )+
                barrier
            }

            unsafe fn on_submit<F>(&self, _: &S, queue: &Arc<Queue>, fence: F)
                                -> SubmitInfo
                where F: FnMut() -> Arc<Fence>
            {
                unimplemented!()
            }
        }

        tuple_impl!($($rest),+);
    );

    ($first:ident) => ();
}

tuple_impl!(A, C, D, E, G, H, J, K, M, N, O, P, Q, R, T, U, V, W, X, Y, Z);
