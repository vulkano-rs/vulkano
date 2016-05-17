// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::Range;
use std::sync::Arc;

use buffer::sys::UnsafeBuffer;
use command_buffer::Submission;
use memory::Content;
use sync::PipelineBarrier;
use sync::Semaphore;

/// Trait for buffer objects that can be used for GPU commands.
pub unsafe trait Buffer: 'static + Send + Sync {
    /// State of the buffer during the construction of a command buffer.
    type CbConstructionState;
    /// State of the buffer in a command buffer.
    type SyncState;

    /// Returns the inner buffer.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    ///
    /// Two different implementations of the `Buffer` trait must never return the same unsafe
    /// buffer. TODO: too restrictive
    fn inner_buffer(&self) -> &UnsafeBuffer;

    /// Called when the user finishes building a command buffer that uses this buffer.
    ///
    /// This allows the buffer to add a pipeline barrier at the end of the command buffer if needed.
    fn command_buffer_finish(&self, prev_barrier: &mut PipelineBarrier,
                             state: &mut (CbConstructionState, SyncState))
                             -> Option<PipelineBarrier>;

    fn command_buffer_submission(&self, state: &SyncState, submission: &Arc<Submission>);

    #[inline]
    fn size(&self) -> usize {
        self.inner_buffer().size()
    }
}

pub unsafe trait TypedBuffer: Buffer {
    type Content: ?Sized + 'static;

    #[inline]
    fn len(&self) -> usize where Self::Content: Content {
        self.size() / <Self::Content as Content>::indiv_size()
    }
}

pub unsafe trait TransferSourceBuffer: Buffer {
    fn command_buffer_transfer_source(&self, range: Range<usize>,
                                      prev_barrier: &mut PipelineBarrier,
                                      state: &mut Option<(CbConstructionState, SyncState)>)
                                      -> Option<PipelineBarrier>;
}

/// Extension trait for buffers that can be used as a destination for transfer operations.
///
/// This includes filling the buffer, updating the buffer, and copying to the buffer.
pub unsafe trait TransferDestinationBuffer: Buffer {
    /// Called when the user wants to use this buffer for the destination of a transfer operation.
    fn command_buffer_transfer_destination(&self, range: Range<usize>,
                                           prev_barrier: &mut PipelineBarrier,
                                           state: &mut Option<(CbConstructionState, SyncState)>)
                                           -> Option<PipelineBarrier>;
}

pub unsafe trait VertexBuffer: Buffer {
    fn command_buffer_vertex_buffer(&self, range: Range<usize>,
                                    prev_barrier: &mut PipelineBarrier,
                                    state: &mut Option<(CbConstructionState, SyncState)>)
                                    -> Option<PipelineBarrier>;
}
