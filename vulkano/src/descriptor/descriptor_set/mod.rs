// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use command_buffer::std::ResourcesStates;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use descriptor::descriptor::DescriptorDesc;
use device::Queue;
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
mod std;
mod sys;
mod unsafe_layout;

/// Trait for objects that contain a collection of resources that will be accessible by shaders.
///
/// Objects of this type can be passed when submitting a draw command.
pub unsafe trait DescriptorSet {
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
    unsafe fn extract_states_and_transition<L>(&self, num_command: usize, list: &mut L)
                                               -> (Self::State, usize, PipelineBarrierBuilder)
        where L: ResourcesStates;
}

// TODO: re-read docs
pub unsafe trait TrackedDescriptorSetState: ResourcesStates {
    type Finished: TrackedDescriptorSetFinished;

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
