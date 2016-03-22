use std::ops::Range;
use std::sync::Arc;

use buffer::unsafe_buffer::UnsafeBuffer;
use command_buffer::Submission;

pub unsafe trait Buffer {
    /// Returns the inner buffer.
    fn inner(&self) -> &UnsafeBuffer;

    /// Returns whether accessing a range of this buffer should signal a fence.
    fn needs_fence(&self, write: bool, Range<usize>) -> Option<bool>;

    unsafe fn gpu_access(&self, write: bool, Range<usize>, submission: &Arc<Submission>)
                         -> Vec<Arc<Submission>>;
}

pub unsafe trait TypedBuffer: Buffer {
    type Content: ?Sized;
}
