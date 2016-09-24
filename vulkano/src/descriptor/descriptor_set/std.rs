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
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::resources_collection::ResourcesCollection;
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

unsafe impl<R, S> TrackedDescriptorSet<S> for StdDescriptorSet<R>
    where R: ResourcesCollection<S>
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
