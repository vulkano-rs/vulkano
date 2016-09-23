// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Provides an utility trait. It's a helper for descriptor sets and framebuffer.
// TODO: better documentation

use std::sync::Arc;

use buffer::traits::TrackedBuffer;
use command_buffer::std::ResourcesStates;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use device::Queue;
use image::traits::TrackedImage;
use sync::AccessFlagBits;
use sync::Fence;
use sync::PipelineStages;

// TODO: re-read docs
/// Collection of tracked resources. Makes it possible to treat multiple buffers and images as one.
pub unsafe trait ResourcesCollection {
    type State;
    type Finished;

    /// Extracts the states relevant to the buffers and images contained in the descriptor set.
    /// Then transitions them to the right state.
    // TODO: must return a Result if multiple elements conflict with one another
    unsafe fn extract_states_and_transition<L>(&self, num_command: usize, list: &mut L)
                                               -> (Self::State, usize, PipelineBarrierBuilder)
        where L: ResourcesStates;
        
    #[inline]
    unsafe fn extract_buffer_state<B>(&self, _: &mut Self::State, buffer: &B) -> Option<B::CommandListState>
        where B: TrackedBuffer;

    #[inline]
    unsafe fn extract_image_state<I>(&self, _: &mut Self::State, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage;

    unsafe fn finish(&self, Self::State) -> (Self::Finished, PipelineBarrierBuilder);

    // TODO: write docs
    unsafe fn on_submit<F>(&self, &Self::Finished, queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo
        where F: FnMut() -> Arc<Fence>;
}

unsafe impl ResourcesCollection for () {
    type State = ();
    type Finished = ();

    #[inline]
    unsafe fn extract_states_and_transition<L>(&self, _num_command: usize, _list: &mut L)
                                               -> ((), usize, PipelineBarrierBuilder)
        where L: ResourcesStates
    {
        ((), 0, PipelineBarrierBuilder::new())
    }

    #[inline]    
    unsafe fn extract_buffer_state<B>(&self, _: &mut (), buffer: &B) -> Option<B::CommandListState>
        where B: TrackedBuffer
    {
        None
    }

    #[inline]
    unsafe fn extract_image_state<I>(&self, _: &mut (), image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        None
    }

    #[inline]
    unsafe fn finish(&self, _: ()) -> ((), PipelineBarrierBuilder) {
        ((), PipelineBarrierBuilder::new())
    }

    unsafe fn on_submit<F>(&self, _: &(), queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        unimplemented!()        // FIXME:
    }
}

pub struct Buf<B> {
    pub buffer: B,
    pub offset: usize,
    pub size: usize,
    pub write: bool,
    pub stage: PipelineStages,
    pub access: AccessFlagBits,
}

unsafe impl<B> ResourcesCollection for Buf<B>
    where B: TrackedBuffer
{
    type State = B::CommandListState;
    type Finished = B::FinishedState;

    #[inline]
    unsafe fn extract_states_and_transition<L>(&self, num_command: usize, list: &mut L)
                                               -> (Self::State, usize, PipelineBarrierBuilder)
        where L: ResourcesStates
    {
        /*let (mut next_state, next_loc, mut next_builder) = {
            self.rest.extract_states_and_transition(num_command, list)
        };

        let my_buf_state = ResourcesStatesJoin(list, &mut next_state)
                                .extract_buffer_state(&self.buffer)
                                .unwrap_or(self.buffer.initial_state());
        
        let (my_buf_state, my_builder) = {
            my_buf_state.transition(num_command, self.buffer.inner(), self.offset, self.size,
                                    self.write, self.stage, self.access)
        };

        // TODO: return Err instead
        assert!(my_builder.as_ref().map(|b| b.after_command_num).unwrap_or(0) <= num_command);
        let command_num = cmp::max(my_builder.as_ref().map(|b| b.after_command_num).unwrap_or(0), next_loc);

        if let Some(my_builder) = my_builder {
            next_builder.add_buffer_barrier_request(self.buffer.inner(), my_builder);
        }

        (BufState(Some(my_buf_state), next_state), command_num, next_builder)*/
        unimplemented!()
    }

    #[inline]
    unsafe fn extract_buffer_state<Ob>(&self, _: &mut Self::State, buffer: &Ob) -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        // TODO:
        unimplemented!()
    }

    #[inline]
    unsafe fn extract_image_state<I>(&self, _: &mut Self::State, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        // TODO:
        unimplemented!()
    }

    #[inline]
    unsafe fn finish(&self, _: Self::State) -> (Self::Finished, PipelineBarrierBuilder) {
        // TODO:
        unimplemented!()
    }

    unsafe fn on_submit<F>(&self, _: &Self::Finished, queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        unimplemented!()        // FIXME:
    }
}

macro_rules! tuple_impl {
    ($first:ident, $($rest:ident),+) => (
        unsafe impl<$first, $($rest),+> ResourcesCollection for ($first $(, $rest)+)
            where $first: ResourcesCollection, $($rest: ResourcesCollection),+
        {
            type State = (<$first as ResourcesCollection>::State,
                          <($($rest),+) as ResourcesCollection>::State);
            type Finished = (<$first as ResourcesCollection>::Finished,
                             <($($rest),+) as ResourcesCollection>::Finished);

            #[inline]
            unsafe fn extract_states_and_transition<L>(&self, _num_command: usize, _list: &mut L)
                                                    -> (Self::State, usize, PipelineBarrierBuilder)
                where L: ResourcesStates
            {
                unimplemented!()
            }

            #[inline]    
            unsafe fn extract_buffer_state<B>(&self, _: &mut Self::State, buffer: &B) -> Option<B::CommandListState>
                where B: TrackedBuffer
            {
                unimplemented!()
            }

            #[inline]
            unsafe fn extract_image_state<I>(&self, _: &mut Self::State, image: &I) -> Option<I::CommandListState>
                where I: TrackedImage
            {
                unimplemented!()
            }

            #[inline]
            unsafe fn finish(&self, _: Self::State) -> (Self::Finished, PipelineBarrierBuilder) {
                unimplemented!()
            }

            unsafe fn on_submit<F>(&self, _: &Self::Finished, queue: &Arc<Queue>, fence: F)
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

tuple_impl!(A, C, D, E, G, H, J, K, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z);
