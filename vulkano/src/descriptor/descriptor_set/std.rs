// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::TrackedBuffer;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::TrackedDescriptorSet;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::sys::UnsafeDescriptorSet;
use device::Queue;
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
    pub offset: usize,
    pub size: usize,
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
        unimplemented!()
    }

    #[inline]
    unsafe fn finish(&self, in_s: &mut S, out: &mut S) -> PipelineBarrierBuilder {
        // TODO:
        unimplemented!()
    }

    unsafe fn on_submit<F>(&self, _: &S, queue: &Arc<Queue>, fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        unimplemented!()        // FIXME:
    }
}

macro_rules! tuple_impl {
    ($first:ident, $($rest:ident),+) => (
        unsafe impl<S, $first, $($rest),+> StdDescriptorSetResourcesCollection<S> for ($first $(, $rest)+)
            where $first: StdDescriptorSetResourcesCollection<S>,
                  $($rest: StdDescriptorSetResourcesCollection<S>),+
        {
            #[inline]
            unsafe fn transition(&self, states: &mut S, _num_command: usize)
                                 -> (usize, PipelineBarrierBuilder)
            {
                unimplemented!()
            }

            #[inline]
            unsafe fn finish(&self, _: &mut S, _: &mut S) -> PipelineBarrierBuilder {
                unimplemented!()
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
