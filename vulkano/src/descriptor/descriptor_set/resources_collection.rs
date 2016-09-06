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

use std::cmp;
use std::iter::Empty;
use std::sync::Arc;

use buffer::traits::CommandListState as BufferCommandListState;
use buffer::traits::TrackedBuffer;
use command_buffer::std::ResourcesStates;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use device::Queue;
use image::traits::TrackedImage;
use sync::AccessFlagBits;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

// TODO: re-read docs
/// Collection of tracked resources. Makes it possible to treat multiple buffers and images as one.
pub unsafe trait ResourcesCollection {
    type State: ResourcesCollectionState<Finished = Self::Finished>;
    type Finished: ResourcesCollectionFinished;

    /// Extracts the states relevant to the buffers and images contained in the descriptor set.
    /// Then transitions them to the right state.
    // TODO: must return a Result if multiple elements conflict with one another
    unsafe fn extract_states_and_transition<L>(&self, list: &mut L, num_command: usize)
                                               -> (Self::State, usize, PipelineBarrierBuilder)
        where L: ResourcesStates;
}

// TODO: re-read docs
pub unsafe trait ResourcesCollectionState: ResourcesStates {
    type Finished: ResourcesCollectionFinished;

    unsafe fn finish(self) -> (Self::Finished, PipelineBarrierBuilder);
}

// TODO: re-read docs
pub unsafe trait ResourcesCollectionFinished {
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

pub struct End;

unsafe impl ResourcesCollection for End {
    type State = End;
    type Finished = End;

    #[inline]
    unsafe fn extract_states_and_transition<L>(&self, _list: &mut L, _num_command: usize)
                                               -> (End, usize, PipelineBarrierBuilder)
        where L: ResourcesStates
    {
        (End, 0, PipelineBarrierBuilder::new())
    }
}

unsafe impl ResourcesStates for End {
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

unsafe impl ResourcesCollectionState for End {
    type Finished = End;

    #[inline]
    unsafe fn finish(self) -> (End, PipelineBarrierBuilder) {
        (End, PipelineBarrierBuilder::new())
    }
}

unsafe impl ResourcesCollectionFinished for End {
    type SemaphoresWaitIterator = Empty<(Arc<Semaphore>, PipelineStages)>;
    type SemaphoresSignalIterator = Empty<Arc<Semaphore>>;

    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        unimplemented!()        // FIXME:
    }
}

pub struct Buf<B, N> {
    pub buffer: B,
    pub offset: usize,
    pub size: usize,
    pub write: bool,
    pub stage: PipelineStages,
    pub access: AccessFlagBits,
    pub rest: N,
}

unsafe impl<B, N> ResourcesCollection for Buf<B, N>
    where B: TrackedBuffer, N: ResourcesCollection
{
    type State = BufState<B::CommandListState, N::State>;
    type Finished = BufFinished<B::FinishedState, N::Finished>;

    #[inline]
    unsafe fn extract_states_and_transition<L>(&self, list: &mut L, num_command: usize)
                                               -> (Self::State, usize, PipelineBarrierBuilder)
        where L: ResourcesStates
    {
        let (mut next_state, next_loc, mut next_builder) = {
            self.rest.extract_states_and_transition(list, num_command)
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

        (BufState(Some(my_buf_state), next_state), command_num, next_builder)
    }
}

pub struct BufState<B, N>(pub Option<B>, pub N);

unsafe impl<B, N> ResourcesCollectionState for BufState<B, N>
    where B: BufferCommandListState, N: ResourcesCollectionState
{
    type Finished = BufFinished<B::FinishedState, N::Finished>;

    #[inline]
    unsafe fn finish(self) -> (Self::Finished, PipelineBarrierBuilder) {
        // TODO:
        unimplemented!()
    }
}

unsafe impl<B, N> ResourcesStates for BufState<B, N> {
    #[inline]
    unsafe fn extract_buffer_state<Ob>(&mut self, buffer: &Ob) -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        // TODO:
        unimplemented!()
    }

    #[inline]
    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        // TODO:
        unimplemented!()
    }
}

pub struct BufFinished<B, N>(pub Option<B>, pub N);

unsafe impl<B, N> ResourcesCollectionFinished for BufFinished<B, N> {
    type SemaphoresWaitIterator = Empty<(Arc<Semaphore>, PipelineStages)>;
    type SemaphoresSignalIterator = Empty<Arc<Semaphore>>;

    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        unimplemented!()        // FIXME:
    }
}

/// Joins two resource states together.
pub struct ResourcesStatesJoin<'a, 'b, A: 'a , B: 'b>(pub &'a mut A, pub &'b mut B);

unsafe impl<'a, 'b, A: 'a , B: 'b> ResourcesStates for ResourcesStatesJoin<'a, 'b, A, B>
    where A: ResourcesStates, B: ResourcesStates
{
    #[inline]
    unsafe fn extract_buffer_state<Ob>(&mut self, buffer: &Ob) -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        if let Some(s) = self.1.extract_buffer_state(buffer) {
            return Some(s);
        }

        self.0.extract_buffer_state(buffer)
    }

    #[inline]
    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        if let Some(s) = self.1.extract_image_state(image) {
            return Some(s);
        }

        self.0.extract_image_state(image)
    }
}
