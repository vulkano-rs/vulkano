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
use descriptor::descriptor::DescriptorDesc;
use device::Queue;
use image::traits::TrackedImage;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

pub use self::collection::DescriptorSetsCollection;
pub use self::pool::DescriptorPool;
pub use self::sys::UnsafeDescriptorSet;
pub use self::sys::DescriptorWrite;
pub use self::unsafe_layout::UnsafeDescriptorSetLayout;

pub mod collection;
pub mod resources_collection;

mod pool;
mod sys;
mod unsafe_layout;

/// Trait for objects that contain a collection of resources that will be accessible by shaders.
///
/// Objects of this type can be passed when submitting a draw command.
// TODO: remove Send + Sync + 'static
pub unsafe trait DescriptorSet: 'static + Send + Sync {
    /// Returns the inner `UnsafeDescriptorSet`.
    fn inner(&self) -> &UnsafeDescriptorSet;
}

/// Trait for objects that describe the layout of the descriptors of a set.
pub unsafe trait DescriptorSetDesc {
    /// Iterator that describes individual descriptors.
    type Iter: ExactSizeIterator<Item = DescriptorDesc>;

    /// Describes the layout of the descriptors of the pipeline.
    fn desc(&self) -> Self::Iter;
}

// TODO: re-read docs
/// Extension trait for descriptor sets so that it can be used with the standard commands list
/// interface.
pub unsafe trait TrackedDescriptorSet: DescriptorSet {
    type State: TrackedDescriptorSetState<Finished = Self::Finished>;
    type Finished: TrackedDescriptorSetFinished;

    /// Extracts the states relevant to the buffers and images contained in the descriptor set.
    /// Then transitions them to the right state.
    unsafe fn extract_states_and_transition<L>(&self, list: &mut L)
                                               -> (Self::State, usize, PipelineBarrierBuilder)
        where L: ResourcesStates;
}

// TODO: re-read docs
pub unsafe trait TrackedDescriptorSetState: ResourcesStates {
    type Finished: TrackedDescriptorSetFinished;

    /// Extracts the state of a buffer of the descriptor set, or `None` if the buffer isn't in
    /// the descriptor set.
    ///
    /// Whether the buffer passed as parameter is the same as the one in the descriptor set must be
    /// determined with the `is_same` method of `TrackedBuffer`.
    ///
    /// # Panic
    ///
    /// - Panics if the state of that buffer has already been previously extracted.
    ///
    unsafe fn extract_buffer_state<B>(&mut self, buffer: &B) -> Option<B::CommandListState>
        where B: TrackedBuffer;

    /// Returns the state of an image, or `None` if the image isn't in the descriptor set.
    ///
    /// See the description of `extract_buffer_state`.
    ///
    /// # Panic
    ///
    /// - Panics if the state of that image has already been previously extracted.
    ///
    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage;

    /// Turns the object into a `TrackedDescriptorSetFinished`. All the buffers and images whose
    /// state hasn't been extracted must be have `finished()` called on them as well.
    ///
    /// The function returns a pipeline barrier to append at the end of the command buffer.
    unsafe fn finish(self) -> (Self::Finished, PipelineBarrierBuilder);
}

// TODO: re-read docs
pub unsafe trait TrackedDescriptorSetFinished {
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
