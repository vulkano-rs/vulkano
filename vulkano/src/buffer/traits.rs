// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::any::Any;
use std::sync::Arc;

use buffer::sys::UnsafeBuffer;
use device::Queue;
use image::Image;
use memory::Content;

use sync::AccessFlagBits;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

use VulkanObject;

// TODO: remove 'static + Send + Sync
pub unsafe trait Buffer: 'static + Send + Sync {
    /// Returns the inner buffer.
    fn inner(&self) -> &UnsafeBuffer;

    #[inline]
    fn size(&self) -> usize {
        self.inner().size()
    }
}

/// Extension trait for `Buffer`. Types that implement this can be used in a `StdCommandBuffer`.
///
/// Each buffer and image used in a `StdCommandBuffer` have an associated state which is
/// represented by the `CommandListState` associated type of this trait. You can make multiple
/// buffers or images share the same state by making `is_same` return true.
pub unsafe trait TrackedBuffer: Buffer {
    /// State of the buffer in a list of commands.
    ///
    /// The `Any` bound is here for stupid reasons, sorry.
    // TODO: remove Any bound
    type CommandListState: Any + CommandListState<FinishedState = Self::FinishedState>;
    /// State of the buffer in a finished list of commands.
    type FinishedState: CommandBufferState;

    /// Returns true if TODO.
    ///
    /// If `is_same` returns true, then the type of `CommandListState` must be the same as for the
    /// other buffer. Otherwise a panic will occur.
    #[inline]
    fn is_same_buffer<B>(&self, other: &B) -> bool where B: Buffer {
        self.inner().internal_object() == other.inner().internal_object()
    }

    /// Returns true if TODO.
    ///
    /// If `is_same` returns true, then the type of `CommandListState` must be the same as for the
    /// other image. Otherwise a panic will occur.
    #[inline]
    fn is_same_image<I>(&self, other: &I) -> bool where I: Image {
        false
    }

    /// Returns the state of the buffer when it has not yet been used.
    fn initial_state(&self) -> Self::CommandListState;
}

/// Trait for objects that represent the state of a slice of the buffer in a list of commands.
pub trait CommandListState {
    type FinishedState: CommandBufferState;

    /// Returns a new state that corresponds to the moment after a slice of the buffer has been
    /// used in the pipeline. The parameters indicate in which way it has been used.
    ///
    /// If the transition should result in a pipeline barrier, then it must be returned by this
    /// function.
    // TODO: what should be the behavior if `num_command` is equal to the `num_command` of a
    // previous transition?
    fn transition(self, num_command: usize, buffer: &UnsafeBuffer, offset: usize, size: usize,
                  write: bool, stage: PipelineStages, access: AccessFlagBits)
                  -> (Self, Option<PipelineBarrierRequest>)
        where Self: Sized;

    /// Function called when the command buffer builder is turned into a real command buffer.
    ///
    /// This function can return an additional pipeline barrier that will be applied at the end
    /// of the command buffer.
    fn finish(self) -> (Self::FinishedState, Option<PipelineBarrierRequest>);
}

/// Requests that a pipeline barrier is created.
pub struct PipelineBarrierRequest {
    /// The number of the command after which the barrier should be placed. Must usually match
    /// the number that was passed to the previous call to `transition`, or 0 if the buffer hasn't
    /// been used yet.
    pub after_command_num: usize,

    /// The source pipeline stages of the transition.
    pub source_stage: PipelineStages,

    /// The destination pipeline stages of the transition.
    pub destination_stages: PipelineStages,

    /// If true, the pipeliner barrier is by region. There is literaly no reason to pass `false`
    /// here, but it is included just in case.
    pub by_region: bool,

    /// An optional memory barrier. See the docs of `PipelineMemoryBarrierRequest`.
    pub memory_barrier: Option<PipelineMemoryBarrierRequest>,
}

/// Requests that a memory barrier is created as part of the pipeline barrier.
///
/// By default, a pipeline barrier only guarantees that the source operations are executed before
/// the destination operations, but it doesn't make memory writes made by source operations visible
/// to the destination operations. In order to make so, you have to add a memory barrier.
///
/// The memory barrier always concerns the buffer that is currently being processed. You can't add
/// a memory barrier that concerns another resource.
pub struct PipelineMemoryBarrierRequest {
    /// Offset of start of the range to flush.
    pub offset: usize,
    /// Size of the range to flush.
    pub size: usize,
    /// Source accesses.
    pub source_access: AccessFlagBits,
    /// Destination accesses.
    pub destination_access: AccessFlagBits,
}

/// Trait for objects that represent the state of the buffer in a command buffer.
pub trait CommandBufferState {
    /// Called right before the command buffer is submitted.
    // TODO: function should be unsafe because it must be guaranteed that a cb is submitted
    fn on_submit<B, F>(&self, buffer: &B, queue: &Arc<Queue>, fence: F) -> SubmitInfos
        where B: Buffer, F: FnOnce() -> Arc<Fence>;
}

pub struct SubmitInfos {
    pub pre_semaphore: Option<(Arc<Semaphore>, PipelineStages)>,
    pub post_semaphore: Option<Arc<Semaphore>>,
    pub pre_barrier: Option<PipelineBarrierRequest>,
    pub post_barrier: Option<PipelineBarrierRequest>,
}

unsafe impl<B> Buffer for Arc<B> where B: Buffer {
    #[inline]
    fn inner(&self) -> &UnsafeBuffer {
        (**self).inner()
    }

    #[inline]
    fn size(&self) -> usize {
        (**self).size()
    }
}

unsafe impl<B> TrackedBuffer for Arc<B> where B: TrackedBuffer, Arc<B>: Buffer {
    type CommandListState = B::CommandListState;
    type FinishedState = B::FinishedState;

    #[inline]
    fn is_same_buffer<Bo>(&self, other: &Bo) -> bool where Bo: Buffer {
        (**self).is_same_buffer(other)
    }

    #[inline]
    fn is_same_image<I>(&self, other: &I) -> bool where I: Image {
        (**self).is_same_image(other)
    }

    #[inline]
    fn initial_state(&self) -> Self::CommandListState {
        (**self).initial_state()
    }
}

pub unsafe trait TypedBuffer: Buffer {
    type Content: ?Sized + 'static;

    #[inline]
    fn len(&self) -> usize where Self::Content: Content {
        self.size() / <Self::Content as Content>::indiv_size()
    }
}
