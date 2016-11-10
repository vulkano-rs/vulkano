// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use command_buffer::cmd::CommandsListSink;
use command_buffer::SubmitInfo;
use command_buffer::StatesManager;
use command_buffer::sys::PipelineBarrierBuilder;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::DescriptorSetDesc;
use descriptor::descriptor_set::TrackedDescriptorSet;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use device::Queue;
use sync::Fence;

/// A collection of descriptor set objects.
pub unsafe trait DescriptorSetsCollection {
    /// Returns the number of sets in the collection. Includes possibly empty sets.
    ///
    /// In other words, this should be equal to the highest set number plus one.
    fn num_sets(&self) -> usize;

    /// Returns the descriptor set with the given id. Returns `None` if the set is empty.
    fn descriptor_set(&self, set: usize) -> Option<&UnsafeDescriptorSet>;

    /// Returns the number of descriptors in the set. Includes possibly empty descriptors.
    ///
    /// Returns `None` if the set is out of range.
    fn num_bindings_in_set(&self, set: usize) -> Option<usize>;

    /// Returns the descriptor for the given binding of the given set.
    ///
    /// Returns `None` if out of range.
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc>;
}

/// Extension trait for a descriptor sets collection so that it can be used with the standard
/// commands list interface.
pub unsafe trait TrackedDescriptorSetsCollection<States = StatesManager>: DescriptorSetsCollection {
    fn add_transition<'a>(&'a self, &mut CommandsListSink<'a>);

    /// Extracts the states relevant to the buffers and images contained in the descriptor sets.
    /// Then transitions them to the right state and returns a pipeline barrier to insert as part
    /// of the transition. The `usize` is the location of the barrier.
    #[deprecated]
    unsafe fn transition(&self, states: &mut States) -> (usize, PipelineBarrierBuilder);

    /// Turns the object into a `TrackedDescriptorSetsCollectionFinished`. All the buffers and
    /// images whose state hasn't been extracted must be have `finished()` called on them as well.
    ///
    /// The function returns a pipeline barrier to append at the end of the command buffer.
    #[deprecated]
    unsafe fn finish(&self, in_s: &mut States, out: &mut States) -> PipelineBarrierBuilder;

    // TODO: write docs
    unsafe fn on_submit<F>(&self, state: &States, queue: &Arc<Queue>, fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>;
}

unsafe impl DescriptorSetsCollection for () {
    #[inline]
    fn num_sets(&self) -> usize {
        0
    }

    #[inline]
    fn descriptor_set(&self, set: usize) -> Option<&UnsafeDescriptorSet> {
        None
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        None
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        None
    }
}

unsafe impl<S> TrackedDescriptorSetsCollection<S> for () {
    #[inline]
    fn add_transition<'a>(&'a self, _: &mut CommandsListSink<'a>) {
    }

    #[inline]
    unsafe fn transition(&self, _: &mut S) -> (usize, PipelineBarrierBuilder) {
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

unsafe impl<T> DescriptorSetsCollection for T
    where T: DescriptorSet + DescriptorSetDesc
{
    #[inline]
    fn num_sets(&self) -> usize {
        1
    }

    #[inline]
    fn descriptor_set(&self, set: usize) -> Option<&UnsafeDescriptorSet> {
        match set {
            0 => Some(self.inner()),
            _ => None
        }
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        unimplemented!()
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        unimplemented!()
    }
}

// TODO: we can't be generic over the State because we get a conflicting implementation :-/
unsafe impl<T> TrackedDescriptorSetsCollection for T where T: TrackedDescriptorSet + DescriptorSetDesc /* TODO */ {
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        self.add_transition(sink);
    }

    #[inline]
    unsafe fn transition(&self, states: &mut StatesManager) -> (usize, PipelineBarrierBuilder) {
        TrackedDescriptorSet::transition(self, states, 0 /* FIXME */)
    }

    #[inline]
    unsafe fn finish(&self, i: &mut StatesManager, o: &mut StatesManager) -> PipelineBarrierBuilder {
        TrackedDescriptorSet::finish(self, i, o)
    }

    #[inline]
    unsafe fn on_submit<F>(&self, states: &StatesManager, queue: &Arc<Queue>, fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        TrackedDescriptorSet::on_submit(self, states, queue, fence)
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)*) => (
        unsafe impl<$first$(, $others)*> DescriptorSetsCollection for ($first, $($others),*)
            where $first: DescriptorSet + DescriptorSetDesc
                  $(, $others: DescriptorSet + DescriptorSetDesc)*
        {
            #[inline]
            fn num_sets(&self) -> usize {
                #![allow(non_snake_case)]
                1 $( + {let $others=0;1})*
            }

            #[inline]
            fn descriptor_set(&self, mut set: usize) -> Option<&UnsafeDescriptorSet> {
                #![allow(non_snake_case)]
                #![allow(unused_mut)]       // For the `set` parameter.

                if set == 0 {
                    return Some(self.0.inner());
                }

                let &(_, $(ref $others,)*) = self;

                $(
                    set -= 1;
                    if set == 0 {
                        return Some($others.inner());
                    }
                )*

                None
            }

            #[inline]
            fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                unimplemented!()
            }

            #[inline]
            fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
                unimplemented!()
            }
        }

        unsafe impl<$first$(, $others)*, St> TrackedDescriptorSetsCollection<St> for ($first, $($others),*)
            where $first: TrackedDescriptorSet<St> + DescriptorSetDesc /* TODO */
                  $(, $others: TrackedDescriptorSet<St> + DescriptorSetDesc /* TODO */)*
        {
            #[inline]
            fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
                #![allow(non_snake_case)]
                let &(ref $first, $(ref $others),*) = self;
                $first.add_transition(sink);
                $(
                    $others.add_transition(sink);
                )*
            }

            #[inline]
            unsafe fn transition(&self, states: &mut St) -> (usize, PipelineBarrierBuilder) {
                unimplemented!()
            }

            #[inline]
            unsafe fn finish(&self, i: &mut St, o: &mut St) -> PipelineBarrierBuilder {
                unimplemented!()
            }

            #[inline]
            unsafe fn on_submit<Fe>(&self, states: &St, queue: &Arc<Queue>, fence: Fe) -> SubmitInfo
                where Fe: FnMut() -> Arc<Fence>
            {
                unimplemented!()
            }
        }

        impl_collection!($($others),*);
    );

    () => ();
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
