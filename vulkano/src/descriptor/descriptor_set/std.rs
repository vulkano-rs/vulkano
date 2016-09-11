// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::traits::TrackedBuffer;
use command_buffer::std::ResourcesStates;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::TrackedDescriptorSet;
use descriptor::descriptor_set::TrackedDescriptorSetState;
use descriptor::descriptor_set::TrackedDescriptorSetFinished;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::resources_collection::ResourcesCollection;
use descriptor::descriptor_set::resources_collection::ResourcesCollectionState;
use descriptor::descriptor_set::resources_collection::ResourcesCollectionFinished;
use descriptor::descriptor_set::sys::UnsafeDescriptorSet;
use device::Queue;
use image::traits::TrackedImage;
use sync::Fence;

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

unsafe impl<R> TrackedDescriptorSet for StdDescriptorSet<R>
    where R: ResourcesCollection
{
    type State = StdDescriptorSetState<R>;
    type Finished = StdDescriptorSetFinishedState<R>;

    #[inline]
    unsafe fn extract_states_and_transition<L>(&self, num_command: usize, list: &mut L)
                                               -> (Self::State, usize, PipelineBarrierBuilder)
        where L: ResourcesStates
    {
        let (state, cmd, builder) = self.resources.extract_states_and_transition(num_command, list);
        let state = StdDescriptorSetState { state: state };
        (state, cmd, builder)
    }
}

pub struct StdDescriptorSetState<R> where R: ResourcesCollection {
    state: R::State,
}

unsafe impl<R> ResourcesStates for StdDescriptorSetState<R>
    where R: ResourcesCollection
{
    #[inline]
    unsafe fn extract_buffer_state<B>(&mut self, buffer: &B) -> Option<B::CommandListState>
        where B: TrackedBuffer
    {
        self.state.extract_buffer_state(buffer)
    }

    #[inline]
    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        self.state.extract_image_state(image)
    }
}

unsafe impl<R> TrackedDescriptorSetState for StdDescriptorSetState<R>
    where R: ResourcesCollection
{
    type Finished = StdDescriptorSetFinishedState<R>;

    #[inline]
    unsafe fn finish(self) -> (Self::Finished, PipelineBarrierBuilder) {
        let (finished, barrier) = self.state.finish();
        let finished = StdDescriptorSetFinishedState { state: finished };
        (finished, barrier)
    }
}

pub struct StdDescriptorSetFinishedState<R> where R: ResourcesCollection {
    state: R::Finished,
}

unsafe impl<R> TrackedDescriptorSetFinished for StdDescriptorSetFinishedState<R>
    where R: ResourcesCollection
{
    type SemaphoresWaitIterator =
        <R::Finished as ResourcesCollectionFinished>::SemaphoresWaitIterator;
    type SemaphoresSignalIterator = 
        <R::Finished as ResourcesCollectionFinished>::SemaphoresSignalIterator;

    #[inline]
    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        self.state.on_submit(queue, fence)
    }
}
