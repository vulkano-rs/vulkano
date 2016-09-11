// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;
use std::iter::Empty as EmptyIter;
use std::option::IntoIter as OptionIntoIter;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

use buffer::traits::TrackedBuffer;
use command_buffer::std::ResourcesStates;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::DescriptorSetDesc;
use device::Queue;
use image::traits::TrackedImage;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

/// A collection of descriptor set objects.
pub unsafe trait DescriptorSetsCollection {
    /// An iterator that produces the list of descriptor set objects contained in this collection.
    type ListIter: ExactSizeIterator<Item = Arc<DescriptorSet>>;

    /// An iterator that produces the description of the list of sets.
    type SetsIter: ExactSizeIterator<Item = Self::DescIter>;

    /// An iterator that produces the description of a set.
    type DescIter: ExactSizeIterator<Item = DescriptorDesc>;

    /// Returns the list of descriptor set objects of this collection.
    fn list(&self) -> Self::ListIter;

    /// Produces a description of the sets, as if it was a layout.
    fn description(&self) -> Self::SetsIter;
}

/// Extension trait for a descriptor sets collection so that it can be used with the standard
/// commands list interface.
pub unsafe trait TrackedDescriptorSetsCollection: DescriptorSetsCollection {
    /// State of the resources inside the collection.
    type State: TrackedDescriptorSetsCollectionState<Finished = Self::Finished>;
    /// Finished state of the resources inside the collection.
    type Finished: TrackedDescriptorSetsCollectionFinished;

    /// Extracts the states relevant to the buffers and images contained in the descriptor sets.
    /// Then transitions them to the right state and returns a pipeline barrier to insert as part
    /// of the transition. The `usize` is the location of the barrier.
    unsafe fn extract_states_and_transition<S>(&self, list: &mut S)
                                               -> (Self::State, usize, PipelineBarrierBuilder)
        where S: ResourcesStates;
}

/// State of the resources inside the collection.
pub unsafe trait TrackedDescriptorSetsCollectionState: ResourcesStates {
    /// Finished state of the resources inside the collection.
    type Finished: TrackedDescriptorSetsCollectionFinished;

    /// Turns the object into a `TrackedDescriptorSetsCollectionFinished`. All the buffers and
    /// images whose state hasn't been extracted must be have `finished()` called on them as well.
    ///
    /// The function returns a pipeline barrier to append at the end of the command buffer.
    unsafe fn finish(self) -> (Self::Finished, PipelineBarrierBuilder);
}

/// Finished state of the resources inside the collection.
pub unsafe trait TrackedDescriptorSetsCollectionFinished {
    /// Iterator that returns the list of semaphores to wait upon before the command buffer is
    /// submitted.
    type SemaphoresWaitIterator: Iterator<Item = (Arc<Semaphore>, PipelineStages)>;

    /// Iterator that returns the list of semaphores to signal after the command buffer has
    /// finished execution.
    type SemaphoresSignalIterator: Iterator<Item = Arc<Semaphore>>;

    // TODO: write docs
    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>;
}

unsafe impl DescriptorSetsCollection for () {
    type ListIter = EmptyIter<Arc<DescriptorSet>>;
    type SetsIter = EmptyIter<EmptyIter<DescriptorDesc>>;
    type DescIter = EmptyIter<DescriptorDesc>;

    #[inline]
    fn list(&self) -> Self::ListIter {
        iter::empty()
    }

    #[inline]
    fn description(&self) -> Self::SetsIter {
        iter::empty()
    }
}

unsafe impl TrackedDescriptorSetsCollection for () {
    type State = EmptyState;
    type Finished = EmptyState;

    #[inline]
    unsafe fn extract_states_and_transition<S>(&self, list: &mut S)
        -> (Self::State, usize, PipelineBarrierBuilder)
        where S: ResourcesStates
    {
        (EmptyState, 0, PipelineBarrierBuilder::new())
    }
}

#[derive(Debug, Copy, Clone)]
pub struct EmptyState;

unsafe impl TrackedDescriptorSetsCollectionState for EmptyState {
    type Finished = EmptyState;

    #[inline]
    unsafe fn finish(self) -> (Self::Finished, PipelineBarrierBuilder) {
        (EmptyState, PipelineBarrierBuilder::new())
    }
}

unsafe impl ResourcesStates for EmptyState {
    #[inline]
    unsafe fn extract_buffer_state<B>(&mut self, buffer: &B) -> Option<B::CommandListState>
        where B: TrackedBuffer
    {
        None
    }

    #[inline]
    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        None
    }
}

unsafe impl TrackedDescriptorSetsCollectionFinished for EmptyState {
    type SemaphoresWaitIterator = EmptyIter<(Arc<Semaphore>, PipelineStages)>;
    type SemaphoresSignalIterator = EmptyIter<Arc<Semaphore>>;

    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        SubmitInfo {
            semaphores_wait: iter::empty(),
            semaphores_signal: iter::empty(),
            pre_pipeline_barrier: PipelineBarrierBuilder::new(),
            post_pipeline_barrier: PipelineBarrierBuilder::new(),
        }
    }
}

// TODO: remove  + 'static + Send + Sync
unsafe impl<'a, T> DescriptorSetsCollection for Arc<T>
    where T: DescriptorSet + DescriptorSetDesc + 'static + Send + Sync
{
    type ListIter = OptionIntoIter<Arc<DescriptorSet>>;
    type SetsIter = OptionIntoIter<Self::DescIter>;
    type DescIter = <T as DescriptorSetDesc>::Iter;

    #[inline]
    fn list(&self) -> Self::ListIter {
        Some(self.clone() as Arc<_>).into_iter()
    }

    #[inline]
    fn description(&self) -> Self::SetsIter {
        Some(self.desc()).into_iter()
    }
}

// TODO: remove  + 'static + Send + Sync
unsafe impl<'a, T> DescriptorSetsCollection for &'a Arc<T>
    where T: DescriptorSet + DescriptorSetDesc + 'static + Send + Sync
{
    type ListIter = OptionIntoIter<Arc<DescriptorSet>>;
    type SetsIter = OptionIntoIter<Self::DescIter>;
    type DescIter = <T as DescriptorSetDesc>::Iter;

    #[inline]
    fn list(&self) -> Self::ListIter {
        Some((*self).clone() as Arc<_>).into_iter()
    }

    #[inline]
    fn description(&self) -> Self::SetsIter {
        Some(self.desc()).into_iter()
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)*) => (
        unsafe impl<'a, $first$(, $others)*> DescriptorSetsCollection for
                                                        (&'a Arc<$first>, $(&'a Arc<$others>),*)
            where $first: DescriptorSet + DescriptorSetDesc + 'static
                  $(, $others: DescriptorSet + DescriptorSetDesc + 'static)*
        {
            type ListIter = VecIntoIter<Arc<DescriptorSet>>;
            type SetsIter = VecIntoIter<Self::DescIter>;
            type DescIter = VecIntoIter<DescriptorDesc>;

            #[inline]
            fn list(&self) -> Self::ListIter {
                #![allow(non_snake_case)]
                let ($first, $($others),*) = *self;

                let list = vec![
                    $first.clone() as Arc<_>,
                    $($others.clone() as Arc<_>),*
                ];

                list.into_iter()
            }

            #[inline]
            fn description(&self) -> Self::SetsIter {
                #![allow(non_snake_case)]
                let ($first, $($others),*) = *self;

                let mut list = Vec::new();
                list.push($first.desc().collect::<Vec<_>>().into_iter());
                $(list.push($others.desc().collect::<Vec<_>>().into_iter());)*
                list.into_iter()
            }
        }

        impl_collection!($($others),*);
    );

    () => ();
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
